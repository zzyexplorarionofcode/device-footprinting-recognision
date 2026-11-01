"""Reuse the project's sklearn MLP (64,32) with capture-held-out model selection."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import hashlib
import joblib
import numpy as np
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from .observation import features_for_flows


@dataclass
class Proxy:
    scaler: StandardScaler
    model: MLPClassifier
    metadata: dict
    frozen: bool = True

    def eval(self):
        # sklearn has no dropout/train mode; predict_proba is its inference-only path.
        self.frozen = True
        return self

    def predict_proba(self, features):
        if not self.frozen:
            raise RuntimeError('Policy training requires a frozen inference proxy')
        return self.model.predict_proba(self.scaler.transform(features))

    def fingerprint(self):
        h = hashlib.sha256()
        for a in [self.scaler.mean_,self.scaler.scale_,*self.model.coefs_,*self.model.intercepts_]:
            h.update(np.asarray(a).tobytes())
        return h.hexdigest()


def train_proxy(train_flows,train_labels,validation_flows,validation_labels,config,*,split_metadata=None,progress=None):
    x,xv = features_for_flows(train_flows,config),features_for_flows(validation_flows,config)
    y,yv = np.asarray(train_labels,int),np.asarray(validation_labels,int)
    classes = np.unique(y)
    if not np.array_equal(classes,np.arange(len(classes))) or not np.array_equal(np.unique(yv),classes):
        raise ValueError('Training and capture-validation must both contain all encoded classes')
    if config['data'].get('expected_classes') not in (None,len(classes)) or len(config['data']['class_names']) != len(classes):
        raise ValueError('Configured class count/name order must match the proxy training data')
    if len(y) != len(x) or len(yv) != len(xv):
        raise ValueError('Label/flow count mismatch')
    cfg,seed = config['proxy'],int(config['seed'])
    scaler = StandardScaler().fit(x)
    x,xv = scaler.transform(x),scaler.transform(xv)
    # "balanced" controls loss weights, never removal of training rows.
    weights = compute_sample_weight('balanced', y) if cfg['balanced'] else None
    model = MLPClassifier(hidden_layer_sizes=tuple(cfg['hidden_layer_sizes']),activation='relu',solver='adam',
                          learning_rate_init=cfg['learning_rate'],alpha=cfg['alpha'],
                          batch_size=min(cfg['batch_size'],len(y)),max_iter=1,early_stopping=False,
                          shuffle=True,random_state=seed)
    best,best_key,history = None,None,[]
    for epoch in range(cfg['epochs']):
        model.partial_fit(x,y,classes=classes,sample_weight=weights)
        score = float(log_loss(yv,model.predict_proba(xv),labels=classes))
        key = (score,epoch)
        history.append({'epoch':epoch,'validation_cross_entropy':score})
        if best_key is None or key < best_key:
            best,best_key = deepcopy(model),key
        if progress:
            progress(f'Proxy epoch {epoch+1}/{cfg["epochs"]}: validation CE={score:.6f}')
    metadata = {'implementation':'sklearn.MLPClassifier; full training data; explicit capture validation',
                'training_protocol':'full-data-class-weight-v1',
                'configuration':deepcopy(cfg),'seed':seed,'classes':classes.tolist(),'input_dimension':x.shape[1],
                'feature_config':deepcopy(config['feature']),'window_length':config['window_length'],
                'class_names':deepcopy(config['data']['class_names']),
                'train_count':len(y),'validation_count':len(yv),'fit_rows':len(y),
                'selection':'minimum validation cross entropy, then epoch','selected_epoch':best_key[1],
                'history':history,'split':deepcopy(split_metadata),'frozen_during_policy_learning':True}
    return Proxy(scaler,best,metadata).eval()


def save_proxy(proxy,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(proxy,path)


def load_proxy(path,config=None,split_metadata=None):
    proxy = joblib.load(path)  # Only load trusted locally generated artifacts.
    if not isinstance(proxy,Proxy):
        raise ValueError('Not a Securitas-adapted proxy')
    if proxy.metadata.get('training_protocol') != 'full-data-class-weight-v1':
        raise ValueError('Proxy predates full-data training; retrain the proxy and policy')
    if config is not None:
        if proxy.metadata['feature_config'] != config['feature'] or proxy.metadata['window_length'] != config['window_length'] or proxy.metadata['class_names'] != config['data']['class_names']:
            raise ValueError('Proxy feature/class configuration mismatch')
    if split_metadata is not None and proxy.metadata['split'] != split_metadata:
        raise ValueError('Proxy was trained on different data partitions or smoke subset')
    return proxy.eval()

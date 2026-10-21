import numpy as np
from inference import compute_ece, metric_value, paired_bootstrap, paired_randomization, cluster_bootstrap, holm_adjust

def main():
    y=np.array([0,1,0,1,0,1]); a=np.array([[.9,.1],[.1,.9],[.8,.2],[.2,.8],[.7,.3],[.3,.7]])
    b=np.array([[.6,.4],[.6,.4],[.55,.45],[.55,.45],[.51,.49],[.51,.49]])
    assert compute_ece(y,a)>=0
    assert metric_value('accuracy',y,a,2)==1.0
    r=paired_bootstrap(y,a,b,2,'accuracy',B=100,seed=1); assert r['delta']>0
    p=paired_randomization(y,a,b,2,'accuracy',B=100,seed=2); assert 0<=p<=1
    c=cluster_bootstrap(y,a,b,np.array([0,0,1,1,2,2]),2,'accuracy',B=100,seed=3,fixed29=False); assert c['n_units']==3
    h=holm_adjust([.01,.03,.2]); assert all(0<=x<=1 for x in h)
    print('PHASE 4 SELF-TEST: PASS')
if __name__=='__main__': main()

import numpy as np
import pandas as pd
from math import isfinite

# ---------- Generic tariff ----------
def compute_bill_generic_india(units, city_type='Urban'):
    slabs = [(100, 3.0), (200, 4.5), (float('inf'), 6.5)]
    energy, br = calc_slab_bill(units, slabs)
    fixed = 150 if city_type == 'Urban' else 100
    return int(round(fixed + energy)), br, fixed

def calc_slab_bill(units, slabs):
    remaining, total, breakdown = float(units), 0.0, []
    for limit, rate in slabs:
        take = min(remaining, limit) if isfinite(limit) else remaining
        if take <= 0: break
        cost = take * rate
        breakdown.append({'Units': take, 'Rate': rate, 'Cost': cost})
        total += cost; remaining -= take
    return total, breakdown

# ---------- Tamil Nadu (TANGEDCO Domestic I-A) ----------
def compute_bill_tn(units_monthly, cycle='Monthly'):
    cycle = cycle.lower()
    if cycle.startswith('bi'):
        slabs = [(100, 0.0),(100,2.25),(200,4.5),(100,6.0),(float('inf'),8.0)]
        u = units_monthly*2
        f = 0 if u<=100 else 20 if u<=200 else 30 if u<=500 else 50
        name='Bi-monthly'
    else:
        slabs=[(50,0.0),(50,2.25),(100,4.5),(50,6.0),(float('inf'),8.0)]
        u=units_monthly
        f=0 if u<=50 else 10 if u<=100 else 15 if u<=250 else 25
        name='Monthly'
    e,br=calc_slab_bill(u,slabs)
    t=int(round(e+f))
    return {'total_cycle':t,'equivalent_monthly':int(round(t/(2 if name=='Bi-monthly' else 1))),
            'breakdown':br,'fixed':f,'units_cycle':u,'cycle_name':name}

# ---------- Dependency & device handling ----------
def parse_devices_textarea(text):
    devices=[]
    for line in text.strip().splitlines():
        if not line.strip(): continue
        parts=[p.strip() for p in line.split(',')]
        if len(parts)<2: continue
        name=parts[0]; hours=float(parts[1]) if parts[1].replace('.','',1).isdigit() else 1.0
        always=len(parts)>=3 and parts[2].lower().startswith('y')
        devices.append({'name':name,'hours':hours,'always_on':always})
    return devices

def classify_dependency(device):
    if device['always_on'] or device['hours']>=6: return 'fully-dependent'
    elif device['hours']>=2: return 'partially-dependent'
    return 'non-dependent'

def estimate_units_from_devices(devices):
    power_map={'ac':1500,'fridge':150,'tv':120,'fan':70,
               'washing machine':500,'microwave':1200,'geyser':3000,
               'light':10,'computer':200,'laptop':60}
    total=0
    for d in devices:
        p=100
        for k,v in power_map.items():
            if k in d['name'].lower(): p=v; break
        total+=p*d['hours']
    return max(10,int(total/1000*30))

# ---------- Demo dataset for ML ----------
def compute_bill_generic_india_sample(units, city):
    fixed=100+city*50
    if units<=100: e=units*3
    elif units<=300: e=100*3+(units-100)*4.5
    else: e=100*3+200*4.5+(units-300)*6.5
    return int(round(fixed+e))

def simulate_dataset(n_months=48, seed=42):
    np.random.seed(seed); rows=[]
    for h in range(200):
        people=np.random.randint(1,6); city=np.random.choice([0,1])
        eff=np.round(np.random.uniform(0.65,0.95),2)
        base=np.random.randint(80,300)
        for m in range(1,n_months+1):
            month=(m%12) or 12
            temp=25+10*np.sin((month-1)/12*2*np.pi)+np.random.randn()*2
            ac=1 if temp>28 else 0
            units=base+ac*120+people*10+(30-temp)*-5
            units=int(units/eff+np.random.randint(-40,40))
            f,p,n=np.random.randint(1,5),np.random.randint(0,6),np.random.randint(0,4)
            bill=compute_bill_generic_india_sample(units,city)+np.random.randint(-150,150)
            rows.append([month,temp,units,people,ac,city,eff,f,p,n,bill])
    return pd.DataFrame(rows,columns=['month','avg_temp','units','people','ac','city','efficiency',
                                      'fully_dep_count','partial_dep_count','non_dep_count','bill'])

import csv,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CATALOG=list(csv.reader((ROOT/'catalog.csv').open()))
assert len(CATALOG)==16
cases={r[0]:{'id':r[0],'name':r[1],'shape':list(map(int,r[2:5])),'width':int(r[5])} for r in CATALOG}
rows=[]
for workers,scenario_ids in [(1,list(cases)),(6,['S04','S05','S06','S07'])]:
    timings=list(csv.reader((ROOT/f'time-{workers}.csv').open()))
    counts=list(csv.reader((ROOT/f'memory-{workers}.csv').open()))
    assert len(timings)==4*len(scenario_ids)*14, (workers,len(timings))
    assert len(counts)==4*len(scenario_ids)*6, (workers,len(counts))
    assert all(int(r[1])==workers and int(r[7])>=3 and float(r[6])*int(r[7])>=79.99 for r in timings)
    for dtype in ['F16','BF16','F32','F64']:
        for scenario in scenario_ids:
            policies={}
            for policy in ['false','true']:
                t=[r for r in timings if r[2]==dtype and r[3]==scenario and r[4]==policy]
                c=[r for r in counts if r[2]==dtype and r[3]==scenario and r[4]==policy]
                assert len(t)==7 and {int(r[5]) for r in t}==set(range(7))
                assert len(c)==3 and {int(r[5]) for r in c}==set(range(3))
                assert all(int(r[10])==0 for r in c)
                latencies=[float(r[6]) for r in t]
                peaks=[int(r[6]) for r in c]
                policies[policy]={'ms':statistics.median(latencies),'ms_range':[min(latencies),max(latencies)],'all_ms':latencies,'peak':max(peaks),'peak_range':[min(peaks),max(peaks)],'requests':sorted({int(r[8]) for r in c}),'reallocs':sorted({int(r[9]) for r in c})}
            old,new=policies['false'],policies['true']
            speedup=old['ms']/new['ms']
            reduction=100*(1-new['peak']/old['peak'])
            overlap=max(old['ms_range'][0],new['ms_range'][0])<=min(old['ms_range'][1],new['ms_range'][1])
            rows.append({'workers':workers,'dtype':dtype,**cases[scenario],'before':old,'after':new,'speedup':speedup,'heap_reduction_percent':reduction,'timing_ranges_overlap':overlap})
summary={'cases':list(cases.values()),'rows':rows}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
for workers in [1,6]:
    print(f'WORKERS {workers}')
    for dtype in ['F16','BF16','F32','F64']:
        print(dtype)
        for r in rows:
            if r['workers']!=workers or r['dtype']!=dtype: continue
            a,b=r['before'],r['after']
            print(f'{r["id"]} {a["ms"]:.6f}->{b["ms"]:.6f}ms {r["speedup"]:.2f}x {a["peak"]:,}->{b["peak"]:,}B ({r["heap_reduction_percent"]:.3f}%) peak_ranges={a["peak_range"]}->{b["peak_range"]} timing_overlap={r["timing_ranges_overlap"]}')
print('VERIFIED',len(rows),'matched comparisons; all recorded windows and rounds are present.')

#!/usr/bin/env python3
"""Unit-only regression tests: no live SUMO assertions."""
import importlib, pathlib, sys, tempfile, warnings
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
coarse=importlib.import_module('02_coarse_calibrate')

class FakeTraCI:
    class TraCIException(Exception): pass
    def __init__(self):
        self.vehicle=self
        self.mode=[]
    def setRoutingMode(self,v,m):self.mode.append((v,m))
    def rerouteTraveltime(self,v,currentTravelTimes=True):
        if v=='blocked': raise self.TraCIException('unreachable')
    def getRoute(self,v):return ('sensor_0','link_0_1','sensor_1')
    def getRouteIndex(self,v):return 0

def main():
    with tempfile.TemporaryDirectory() as d:
        root=pathlib.Path(d)
        (root/'abstract.net.xml').write_text('<net/>')
        trip=root/'routes.xml'
        trip.write_text('<?xml version="1.0"?><routes><vType id="passenger" vClass="passenger"/><vehicle id="veh0" type="passenger" depart="0"><route edges="sensor_0 link_0_1 sensor_1"/></vehicle></routes>')
        c=coarse.sumo_command(root,trip,100,False)
        assert '--ignore-route-errors' not in c and '--device.rerouting.mode' not in c
        assert c[c.index('-r')+1]==str(trip)
        c=coarse.sumo_command(root,trip,100,True)
        assert c[c.index('--ignore-route-errors')+1]=='true'
        assert '--device.rerouting.mode' not in c
        routed=pathlib.Path(c[c.index('-r')+1])
        assert routed!=trip and routed.exists()
        assert 'device.rerouting.mode' not in trip.read_text()
        import xml.etree.ElementTree as ET
        tree=ET.parse(routed)
        vtype=tree.getroot().find('vType')
        assert any(p.get('key')=='device.rerouting.mode' and p.get('value')=='8' for p in vtype.findall('param'))
        assert tree.getroot().find('vehicle').attrib['depart']=='0'
        assert coarse.sumo_command(root,trip,100,True)[coarse.sumo_command(root,trip,100,True).index('-r')+1]==str(routed)
    fake=FakeTraCI()
    d=coarse._reroute_vehicles(fake,['ok','blocked'],['link_0_1'])
    assert d=={'attempts':2,'failures':1,'unchanged_blocked_routes':1},d
    assert fake.mode==[('ok',0),('ok',8),('blocked',0),('blocked',8)]
    speed=np.vstack([np.array([[np.nan,np.nan]]),np.tile(np.array([[55.,62.],[45.,np.nan]]),(12,1))])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        q=coarse.stats(speed,65)
        assert not caught, caught
    assert q['mean_speed_mph']>0 and np.isfinite(q['peak_valley_speed_ratio'])
    print('PASS SUMO 1.18 command: no unsupported routing-mode CLI option; baseline strict; closure-safe route vType mode=8; reroute audit; coarse stats')
    print('LIVE SUMO test NOT performed here: server SUMO 1.18 + TraCI required')
if __name__=='__main__':main()

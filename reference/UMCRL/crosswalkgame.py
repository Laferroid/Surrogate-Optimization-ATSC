# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 17:53:33 2021

@author: zhuho
"""

#import time
# import os
# import sys
import optparse
#import cmath
import numpy as np
import random
#import python interface of sumo traci
import traci
import xlrd as xd
# import xlwt as xt
import math
from sumolib import checkBinary
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

def calgeomertry (inputX, inputY, crosswidth, lanewidth, medianwidth, detectLength, waitareaLength, stoplinedistance):
    # ,Bdist2crosswalk,Bdist2curb):
    #   (X,Y)
    #--|--|------1------|-----
    #  S1 5             6
    #--|--|------2------|-----
    #-----|------3------|--|--
    #     |             |  S2
    #-----|------4------|--|--
    
    #     -----D1-----
    #
    #     -----W1-----
    #----------L1-----------
    #
    #
    #
    #----------L2-----------
    #     -----W2-----
    #     -----W3-----
    #----------L3-----------
    #
    #
    #
    #----------L4-----------
    #     -----W4----
    #
    #     -----D2----
    #-(B1)|
    #--|  |
    #--|--|------1------|-----
    #  S1 5             6
    #--|--|------2------|-----
    #-----|------3------|--|--
    #     |             |  S2
    #-----|------4------|--|--
    #                   |  |--
    #                   |(B2)-
    global L1Y,L2Y,L3Y,L4Y,L5X,L6X,S1X,S2X,D1Y,D2Y,W1Y,W2Y,W3Y,W4Y
    # B1X,B1Y,B2X,B2Y
    L1Y=inputY
    L2Y=round(inputY-lanewidth+0.5,1)
    L3Y=round(L2Y-medianwidth,1)
    L4Y=round(L3Y-lanewidth+0.5,1)
    L5X=inputX
    L6X=round(inputX+crosswidth,1)
    S1X=round(L5X-stoplinedistance,1)
    S2X=round(L6X+stoplinedistance,1)
    D1Y=round(L1Y+detectLength,1)
    D2Y=round(L4Y-detectLength,1)
    W1Y=round(L1Y+waitareaLength,1)
    W2Y=round(L2Y-waitareaLength,1)
    W3Y=round(L3Y+waitareaLength,1)
    W4Y=round(L4Y-waitareaLength,1)
    # B1X=round(inputX-Bdist2crosswalk,1)
    # B1Y=round(inputY+Bdist2curb,1)
    # B2X=round(L6X+Bdist2crosswalk,1)
    # B2Y=round(L4Y-Bdist2curb,1)
    
def RandomSenario(pvolume, starttime):
    if pvolume==0:
        plist=[0.01,0.01,0.01,0.01,0.01,0.01]
    elif pvolume==1:
        plist=[0.02,0.02,0.02,0.02,0.02,0.02]
    elif pvolume==2:
        plist=[0.03,0.03,0.03,0.03,0.03,0.03]
    elif pvolume==3:
        plist=[0.06,0.06,0.06,0.06,0.06,0.06]
    elif pvolume==4:
        plist=[0.1,0.1,0.1,0.1,0.1,0.1]
    
    with open("test1v1.rou.xml", "w") as routes:
        print("""<?xml version="1.0" encoding="UTF-8"?>""", file=routes)
    rou_write=open('test1v1.rou.xml','a')
        
    rou_write.write('\n<!-- generated on 2021-07-15 22:50:12 by Eclipse SUMO netedit Version 1.9.1\n')
    rou_write.write('<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/netconvertConfiguration.xsd">\n')
    
    
    rou_write.write('    <processing>\n')
    rou_write.write('        <geometry.min-radius.fix.railways value="false"/>\n')
    rou_write.write('        <geometry.max-grade.fix value="false"/>\n')
    rou_write.write('        <offset.disable-normalization value="true"/>\n')
    rou_write.write('        <lefthand value="false"/>\n')
    rou_write.write('    </processing>\n')
    
    rou_write.write('    <junctions>\n')
    rou_write.write('        <no-internal-links value="true"/>\n')
    rou_write.write('        <no-turnarounds value="true"/>\n')
    rou_write.write('        <junctions.corner-detail value="5"/>\n')
    rou_write.write('        <junctions.limit-turn-speed value="5.5"/>\n')
    rou_write.write('        <rectangular-lane-cut value="false"/>\n')
    rou_write.write('    </junctions>\n')
    
    rou_write.write('    <pedestrian>\n')
    rou_write.write('        <walkingareas value="false"/>\n')
    rou_write.write('    </pedestrian>\n') 
    
    rou_write.write('    <report>\n')
    rou_write.write('        <aggregate-warnings value="5"/>\n')
    rou_write.write('    </report>\n')
    
    rou_write.write('</configuration>\n')
    rou_write.write('-->\n')
    
    rou_write.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
    rou_write.write('    <vType id="H30" length="4.50" minGap="2.00" maxSpeed="8.33" emissionClass="HBEFA3/PC_G_EU4" guiShape="passenger/hatchback" width="2.00" height="1.50" color="77,255,54" carFollowModel="IDM" accel="2" decel="3"/>\n')
    rou_write.write('    <vType id="H40" length="4.50" minGap="2.00" maxSpeed="11.11" guiShape="passenger/hatchback" width="2.00" color="255,161,0" carFollowModel="IDM" accel="2" decel="3"/>\n')
    rou_write.write('    <vType id="S30" length="5.00" minGap="2.00" maxSpeed="8.33" emissionClass="HBEFA3/PC_G_EU4" guiShape="passenger" width="2.00" height="1.50" color="77,255,54" carFollowModel="IDM" accel="2" decel="3"/>\n')
    rou_write.write('    <vType id="S40" length="5.00" minGap="2.00" maxSpeed="11.11" guiShape="passenger" width="2.00" color="255,161,0" carFollowModel="IDM" accel="2" decel="3"/>\n')
    rou_write.write('    <vType id="V30" length="6.00" minGap="2.00" maxSpeed="8.33" emissionClass="HBEFA3/PC_G_EU4" guiShape="passenger/van" width="2.20" height="1.50" color="77,255,54" carFollowModel="IDM" accel="2" decel="3"/>\n')
    rou_write.write('    <vType id="V40" length="6.00" minGap="2.00" maxSpeed="11.11" guiShape="passenger/van" width="2.20" color="255,161,0" carFollowModel="IDM" accel="2" decel="3"/>\n')
    rou_write.write('    <vType id="person1" length="0.74" minGap="0.00" maxSpeed="1.58" guiShape="pedestrian" width="0.40" height="1.70" lcTimeToImpatience="9999" carFollowModel="IDM" tau="0.1"/>\n')
    rou_write.write('    <flow id="person_1" type="person1" begin="0.00" departLane="0" arrivalLane="0" end="7200.00" probability="%a">\n' %(plist[0]))
    rou_write.write('        <route edges="gneE3" color="cyan"/>\n')
    rou_write.write('    </flow>\n')       
    rou_write.write('    <flow id="person_2" type="person1" begin="0.00" departLane="1" arrivalLane="1" end="7200.00" probability="%a">\n' %(plist[1]))
    rou_write.write('        <route edges="gneE3" color="cyan"/>\n')
    rou_write.write('    </flow>\n')
    rou_write.write('    <flow id="person_3" type="person1" begin="0.00" departLane="2" arrivalLane="2" end="7200.00" probability="%a">\n' %(plist[2]))
    rou_write.write('        <route edges="gneE3" color="cyan"/>\n')
    rou_write.write('    </flow>\n')       
    rou_write.write('    <flow id="person_4" type="person1" begin="0.00" departLane="0" arrivalLane="0" end="7200.00" probability="%a">\n' %(plist[3]))
    rou_write.write('        <route edges="gneE0" color="cyan"/>\n')
    rou_write.write('    </flow>\n')    
    rou_write.write('    <flow id="person_5" type="person1" begin="0.00" departLane="1" arrivalLane="1" end="7200.00" probability="%a">\n' %(plist[4]))
    rou_write.write('        <route edges="gneE0" color="cyan"/>\n')
    rou_write.write('    </flow>\n')
    rou_write.write('    <flow id="person_6" type="person1" begin="0.00" departLane="2" arrivalLane="2" end="7200.00" probability="%a">\n' %(plist[5]))
    rou_write.write('        <route edges="gneE0" color="cyan"/>\n')
    rou_write.write('    </flow>\n')
    rou_write.write('    <flow id="flow_1" type="H40" begin="%a" departSpeed="max" end="7200.00" number="1">\n' %(starttime))
    rou_write.write('        <route edges="gneE1" color="cyan"/>\n')
    rou_write.write('    </flow>\n')    
    rou_write.write('</routes>')
    rou_write.close()        

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

def group (prefix,inputlist,outputlist):
    for veh in inputlist:
        if veh.startswith(prefix):
            veh=veh.split(" ")
            outputlist+=veh
            
def vehjudge (rolling,peda,pedp,veha,vehp,vehl):
    vehjudge_v=0
    if rolling==0:            
        if peda==0 and pedp[1]<=L4Y and pedp[1]>=W4Y:
            if veha==90 and vehp[0]-vehl<=pedp[0]:
                vehjudge_v=1
            elif veha==270 and vehp[0]+vehl>=pedp[0]:
                vehjudge_v=1
        elif peda==180 and pedp[1]<=W1Y and pedp[1]>=L1Y:
            if veha==90 and vehp[0]-vehl<=pedp[0]:
                vehjudge_v=1                         
            elif veha==270 and vehp[0]+vehl>=pedp[0]:
                vehjudge_v=1
        elif peda==180 and pedp[1]>=L3Y and pedp[1]<=W3Y:
            if veha==270 and vehp[0]+vehl>=pedp[0]:
                vehjudge_v=1
        elif peda==0 and pedp[1]>=W2Y and pedp[1]<=L2Y:
            if veha==90 and vehp[0]-vehl<=pedp[0]:
                vehjudge_v=1        
    if rolling==1:            
        if peda==0 and pedp[1]<=L4Y and pedp[1]>=W4Y:
            if veha==270 and vehp[0]+vehl>=pedp[0]:
                vehjudge_v=1
        elif peda==180 and pedp[1]<=W1Y and pedp[1]>=L1Y:
            if veha==90 and vehp[0]-vehl<=pedp[0]:
                vehjudge_v=1                        
        elif peda==180 and pedp[1]<=W3Y and pedp[1]>=L3Y:
            if veha==270 and vehp[0]+vehl>=pedp[0]:
                vehjudge_v=1                        
        elif peda==0 and pedp[1]<=L2Y and pedp[1]>=W2Y:
            if veha==90 and vehp[0]-vehl<=pedp[0]:
                vehjudge_v=1              
    return vehjudge_v
        
def pedjudge (peda,pedp,veha,vehp,vehl):
    if veha==90 and vehp[0]<=pedp[0]:
        if peda==0 and pedp[1]<=L1Y and pedp[1]>D2Y:
            return 1 
        elif peda==180 and pedp[1]<=D1Y and pedp[1]>=L2Y:
            return 1
        else:
            return 0
    elif veha==270 and vehp[0]>=pedp[1]:
        if peda==0 and pedp[1]<=L3Y and pedp[1]>D2Y:
            return 1 
        elif peda==180 and pedp[1]<=D1Y and pedp[1]>=L4Y:
            return 1
        else:
            return 0

    
def pedmovement_Rolling_Gap (pedlist, vehlist, reactiontime,timestep,ped_extratime,ped_type,veh_reactiontime,veh_dec,ped_scandist):
    global p1e,p1o,p2e,p2o,d,vehlf0,vehlf180
    #ped_type 1:cautious 2:high priority sense 3:Reckless
    if timestep%(reactiontime*10)==0:
        # veh270=[]
        # veh90=[]          
        # for veh in vehlist:
        #     veha=traci.vehicle.getAngle(veh)
        #     vehp=traci.vehicle.getPosition(veh)
        #     if veha==90 and vehp[0]>L5X:
        #         veh90.append(veh)
        #     elif veha==270 and vehp[0]<L6X:
        #         veh270.append(veh)      
        for pedid in pedlist:
            pedp=traci.vehicle.getPosition(pedid)
            peda=traci.vehicle.getAngle(pedid)
            timelist=globals()['Time%s'%pedid]
            if peda==0:
                Yposition=int((2*round((pedp[1]/2),1)+3)/0.2)
                # obvehlist=veh270.copy()
            elif peda==180:
                Yposition=int((2*round(((-pedp[1])/2),1)+10.2)/0.2)
                # obvehlist=veh90.copy()
            if (timestep/(reactiontime*10))%2==0:
                p1e["%s" %pedid]=0                        
                p2e["%s" %pedid]=0
            elif (timestep/(reactiontime*10))%2==1:
                p1o["%s" %pedid]=0                        
                p2o["%s" %pedid]=0              
            for vehid in vehlist:
                veha=traci.vehicle.getAngle(vehid)
                vehspeed=traci.vehicle.getSpeed(vehid)
                vehp=traci.vehicle.getPosition(vehid)
                vehl=traci.vehicle.getLength(vehid)
                #Bindex_ped=ped_veh_visualindex(pedid,obvehlist,vehid)
                Bindex_ped=0
                dist1=(pow((pedp[0]-vehp[0]),2)+pow((pedp[1]-vehp[1]),2))**0.5
                # vehlead=()
                # vehlead=traci.vehicle.getLeader(vehid)
                # leadindex=0
                # if vehlead:
                #     vehleadp=traci.vehicle.getPosition(vehlead[0])
                #     if veha==90 and vehleadp[0]>=pedp[0]:
                #         leadindex=1
                #     elif veha==270 and vehleadp[0]<=pedp[0]:
                #         leadindex=1
                # else:
                #     leadindex=1       
                leadindex=1
                vehjudge_v=vehjudge(1,peda,pedp,veha,vehp,vehl)          
                if vehjudge_v==1 and Bindex_ped==0 and leadindex==1 and dist1<=ped_scandist:                                         
                    if veha==90:
                        if ped_type==1:
                            if peda==0:
                                Tposition=vehlf0[3]
                                pedct=timelist[Tposition]-timelist[Yposition]
                            elif peda==180:
                                Tposition=vehlf180[1]
                                pedct=timelist[Tposition]-timelist[Yposition]
                            dist6=vehp[0]+(pedct+ped_extratime)*vehspeed
                        elif ped_type==2:
                            dist6=vehp[0]+(vehspeed**2)/(2*veh_dec)+vehspeed*(veh_reactiontime+0.2)+0.000001                                  
                        if dist6>= pedp[0]:
                            if (timestep/(reactiontime*10))%2==0:
                                p1e["%s" %pedid]=1
                            elif (timestep/(reactiontime*10))%2==1:
                                p1o["%s" %pedid]=1
                        elif vehp[0]>pedp[0] and (vehp[0]-vehl)<=pedp[0]:
                            if (timestep/(reactiontime*10))%2==0:
                                p2e["%s" %pedid]=1 
                            elif (timestep/(reactiontime*10))%2==1:
                                p2o["%s" %pedid]=1
                    elif veha==270:                     
                        if ped_type==1:
                            if peda==0:
                                Tposition=vehlf0[1]
                                pedct=timelist[Tposition]-timelist[Yposition]
                            elif peda==180:
                                Tposition=vehlf180[3]
                                pedct=timelist[Tposition]-timelist[Yposition]
                            dist6=vehp[0]-(pedct+ped_extratime)*vehspeed
                        elif ped_type==2:
                            dist6=vehp[0]-(vehspeed**2)/(2*veh_dec)+vehspeed*(veh_reactiontime+0.2)+0.000001                             
                        if dist6<=pedp[0]:
                            if (timestep/(reactiontime*10))%2==0:
                                p1e["%s" %pedid]=1
                            elif (timestep/(reactiontime*10))%2==1:
                                p1o["%s" %pedid]=1
                        elif (vehp[0]+vehl)>=pedp[0] and vehp[0]<pedp[0]:
                            if (timestep/(reactiontime*10))%2==0:
                                p2e["%s" %pedid]=1   
                            elif (timestep/(reactiontime*10))%2==1:
                                p2o["%s" %pedid]=1
    for pedid in pedlist:
        pedp=traci.vehicle.getPosition(pedid)
        peda=traci.vehicle.getAngle(pedid)
        speedlist=globals()['Speed%s'%pedid]
        pedtemp1=0
        pedtemp2=0
        if (timestep/(reactiontime*10))%2==0:
            try:
                p1o["%s" %pedid]
            except:
                pedtemp1=0
            else:
                pedtemp1=p1o["%s" %pedid]
            try:
                p2o["%s" %pedid]
            except:
                pedtemp2=0
            else:
                pedtemp2=p2o["%s" %pedid]
        if (timestep/(reactiontime*10))%2==1:
            try:
                p1e["%s" %pedid]
            except:
                pedtemp1=0
            else:                    
                pedtemp1=p1e["%s" %pedid]
            try:
                p2e["%s" %pedid]
            except:
                pedtemp2=0
            else:
                pedtemp2=p2e["%s" %pedid]
        if pedtemp1==0 and pedtemp2==0:
            if peda==0:
                Yposition=int((2*round((pedp[1]/2),1)+3)/0.2)
                if pedp[1]<=L1Y and pedp[1]>=D2Y:
                    # traci.vehicle.setSpeedMode(pedid,0)
                    traci.vehicle.slowDown(pedid,speedlist[Yposition],0.1)
                elif pedp[1]>L1Y:
                    # traci.vehicle.setSpeedMode(pedid,0)
                    traci.vehicle.slowDown(pedid,speedlist[-1],0.1)
            elif peda==180:
                Yposition=int((2*round(((-pedp[1])/2),1)+10.2)/0.2)
                if pedp[1]<=D1Y and pedp[1]>=L4Y:
                    # traci.vehicle.setSpeedMode(pedid,0)
                    traci.vehicle.slowDown(pedid,speedlist[Yposition],0.1)
                elif pedp[1]<L4Y:
                    # traci.vehicle.setSpeedMode(pedid,0)
                    traci.vehicle.slowDown(pedid,speedlist[-1],0.1)
        elif pedtemp1>0 or pedtemp2>0:
                # traci.vehicle.setSpeedMode(pedid,0)
                traci.vehicle.slowDown(pedid,0,0.1)

def pedmovement_One_Stage (pedlist, vehlist, reactiontime,timestep,ped_extratime,ped_type,veh_reactiontime,veh_dec,ped_scandist):
    global p1e,p1o,p2e,p2o,d,vehlf0,vehlf180
    if timestep%(reactiontime*10)==0:
        # veh270=[]
        # veh90=[]
        # for veh in vehlist:
        #     veha=traci.vehicle.getAngle(veh)
        #     vehp=traci.vehicle.getPosition(veh)
            # if veha==90 and vehp[0]>L5X:
                #veh90.append(veh)
            # elif veha==270 and vehp[0]<L6X:
                #veh270.append(veh)
                
        for pedid in pedlist:
            pedp=traci.vehicle.getPosition(pedid)
            peda=traci.vehicle.getAngle(pedid)
            timelist=globals()['Time%s'%pedid]
            if peda==0:
                Yposition=int((2*round((pedp[1]/2),1)+3)/0.2)
                #obvehlist=veh270.copy()
            elif peda==180:
                Yposition=int((2*round(((-pedp[1])/2),1)+10.2)/0.2)
                #obvehlist=veh90.copy()
            if (timestep/(reactiontime*10))%2==0:
                p1e["%s" %pedid]=0                        
                p2e["%s" %pedid]=0
            elif (timestep/(reactiontime*10))%2==1:
                p1o["%s" %pedid]=0                        
                p2o["%s" %pedid]=0
            for vehid in vehlist:
                veha=traci.vehicle.getAngle(vehid)
                vehspeed=traci.vehicle.getSpeed(vehid)
                vehp=traci.vehicle.getPosition(vehid)
                vehl=traci.vehicle.getLength(vehid)
                dist1=(pow((pedp[0]-vehp[0]),2)+pow((pedp[1]-vehp[1]),2))**0.5
                # vehlead=()
                # vehlead=traci.vehicle.getLeader(vehid)
                # leadindex=0
                # if vehlead:
                #     vehleadp=traci.vehicle.getPosition(vehlead[0])
                #     if veha==90 and vehleadp[0]>=pedp[0]:
                #         leadindex=1
                #     elif veha==270 and vehleadp[0]<=pedp[0]:
                #         leadindex=1
                # else:
                #     leadindex=1                
                vehjudge_v=vehjudge(0,peda,pedp,veha,vehp,vehl)                                 
                #Bindex_ped=ped_veh_visualindex(pedid,obvehlist,vehid)
                Bindex_ped=0
                leadindex=1
                if vehjudge_v==1 and leadindex==1 and Bindex_ped==0 and dist1<=ped_scandist:
                    if veha==90:
                        if ped_type==1:
                            if peda==0:
                                Tposition=vehlf0[3]
                                pedct=timelist[Tposition]-timelist[Yposition]
                            elif peda==180:
                                Tposition=vehlf180[1]
                                pedct=timelist[Tposition]-timelist[Yposition]
                            dist6=vehp[0]+(pedct+ped_extratime)*vehspeed
                        elif ped_type==2:
                            dist6=vehp[0]+(vehspeed**2)/(2*veh_dec)+vehspeed*(veh_reactiontime+0.2)+0.000001                                  
                        if dist6>= pedp[0]:
                            if (timestep/(reactiontime*10))%2==0:
                                p1e["%s" %pedid]=1
                            elif (timestep/(reactiontime*10))%2==1:
                                p1o["%s" %pedid]=1
                        elif vehp[0]>pedp[0] and (vehp[0]-vehl)<=pedp[0]:
                            if (timestep/(reactiontime*10))%2==0:
                                p2e["%s" %pedid]=1 
                            elif (timestep/(reactiontime*10))%2==1:
                                p2o["%s" %pedid]=1                    
                    elif veha==270:                     
                        if ped_type==1:
                            if peda==0:
                                Tposition=vehlf0[1]
                                pedct=timelist[Tposition]-timelist[Yposition]
                            elif peda==180:
                                Tposition=vehlf180[3]
                                pedct=timelist[Tposition]-timelist[Yposition]
                            dist6=vehp[0]-(pedct+ped_extratime)*vehspeed
                        elif ped_type==2:
                            dist6=vehp[0]-(vehspeed**2)/(2*veh_dec)+vehspeed*(veh_reactiontime+0.2)+0.000001                             
                        if dist6<=pedp[0]:
                            if (timestep/(reactiontime*10))%2==0:
                                p1e["%s" %pedid]=1
                            elif (timestep/(reactiontime*10))%2==1:
                                p1o["%s" %pedid]=1
                        elif (vehp[0]+vehl)>=pedp[0] and vehp[0]<pedp[0]:
                            if (timestep/(reactiontime*10))%2==0:
                                p2e["%s" %pedid]=1   
                            elif (timestep/(reactiontime*10))%2==1:
                                p2o["%s" %pedid]=1                                                                 
    for pedid in pedlist:
        pedp=traci.vehicle.getPosition(pedid)
        peda=traci.vehicle.getAngle(pedid)
        speedlist=globals()['Speed%s'%pedid]
        pedtemp1=0
        pedtemp2=0
        if (timestep/(reactiontime*10))%2==0:
            try:
                p1o["%s" %pedid]
            except:
                pedtemp1=0
            else:
                pedtemp1=p1o["%s" %pedid]
            try:
                p2o["%s" %pedid]
            except:
                pedtemp2=0
            else:
                pedtemp2=p2o["%s" %pedid]
        if (timestep/(reactiontime*10))%2==1:
            try:
                p1e["%s" %pedid]
            except:
                pedtemp1=0
            else:                    
                pedtemp1=p1e["%s" %pedid]
            try:
                p2e["%s" %pedid]
            except:
                pedtemp2=0
            else:
                pedtemp2=p2e["%s" %pedid]
        if pedtemp1==0 and pedtemp2==0:
            if peda==0:
                Yposition=int((2*round((pedp[1]/2),1)+3)/0.2)
                if pedp[1]<=L1Y and pedp[1]>=D2Y:
                    traci.vehicle.setSpeedMode(pedid,1)
                    traci.vehicle.slowDown(pedid,speedlist[Yposition],0.1)
                elif pedp[1]>L1Y:
                    traci.vehicle.setSpeedMode(pedid,1)
                    traci.vehicle.slowDown(pedid,speedlist[-1],0.1)
            elif peda==180:
                Yposition=int((2*round(((-pedp[1])/2),1)+10.2)/0.2)
                if pedp[1]<=D1Y and pedp[1]>=L4Y:
                    traci.vehicle.setSpeedMode(pedid,1)
                    traci.vehicle.slowDown(pedid,speedlist[Yposition],0.1)
                elif pedp[1]<L4Y:
                    traci.vehicle.setSpeedMode(pedid,1)
                    traci.vehicle.slowDown(pedid,speedlist[-1],0.1)
        elif pedtemp1>0 or pedtemp2>0:
                traci.vehicle.setSpeedMode(pedid,1)
                traci.vehicle.slowDown(pedid,0,0.1)

def keycreating (vehlist,index,lowersize,uppersize):
    #1:30-->GD 31:54-->LS 55:176-->MS 176:200-->HS
    global d
    if index==1:
        for veh in vehlist:
            if veh not in d.keys():
                d[veh]=len(d)+1
                
    elif index==2:
        for veh in vehlist:
            if veh not in d.keys():
                d[veh]=random.randint(lowersize,uppersize)
            

def vehicleaction(vehlist,action):
    # 还未考虑车辆的延迟
    for vehid in vehlist:
        vehtype=traci.vehicle.getTypeID(vehid)
        maxspeed=float(vehtype[1:3])/3.6
        vehspeed=traci.vehicle.getSpeed(vehid)         
        if action==3:
            traci.vehicle.setSpeed(vehid,maxspeed/2)
            traci.vehicle.setColor(vehid,(255,255,0))
        elif action==1:
            traci.vehicle.setSpeed(vehid,maxspeed)
            traci.vehicle.setColor(vehid,(0,0,139))
        elif action==2:
            traci.vehicle.setSpeed(vehid,vehspeed)
            traci.vehicle.setColor(vehid,(0,250,154))
        elif action==0:
            traci.vehicle.setSpeed(vehid,0)
            traci.vehicle.setColor(vehid,(139,58,58))
        
        
# def judgeture(vehlist,pedlist,step):
#     judge_v=0
#     for veh in vehlist:
#         vehp=traci.vehicle.getPosition(veh)
#         veha=traci.vehicle.getAngle(veh)
#         vehl=traci.vehicle.getLength(veh)
#         vehw=traci.vehicle.getWidth(veh)
#         if veha==90 and vehp[0]>10:
#             judge_v=1
#         elif veha==270 and vehp[0]<-10:
#             judge_v=1
#         else:
#             for ped in pedlist:
#                 pedp=traci.vehicle.getPosition(ped)
#                 if veha==90 and pedp[0]<vehp[0] and pedp[0]>vehp[0]-vehl and pedp[1]>vehp[1]-vehw/2 and pedp[1]<vehp[1]+vehw/2:
#                     judge_v=2
#                 elif veha==270 and pedp[0]>vehp[0] and pedp[0]<vehp[0]+vehl and pedp[1]>vehp[1]-vehw/2 and pedp[1]<vehp[1]+vehw/2:
#                     judge_v=2
#     if judge_v==2:
#         print('ped-hit')
#     return judge_v

def judgeture(vehlist,pedlist,step):
    judge_v=0
    global crashnumber
    for veh in vehlist:
        vehp=traci.vehicle.getPosition(veh)
        veha=traci.vehicle.getAngle(veh)
        vehl=traci.vehicle.getLength(veh)
        # vehw=traci.vehicle.getWidth(veh)
        if veha==90 and vehp[0]>10:
            judge_v=1
        elif veha==270 and vehp[0]<-10:
            judge_v=1
        else:
            for ped in pedlist:
                pedp=traci.vehicle.getPosition(ped)
                if veha==90 and pedp[0]<vehp[0] and pedp[0]>vehp[0]-vehl and pedp[1]>L2Y and pedp[1]<L1Y:
                    judge_v=2
                elif veha==270 and pedp[0]>vehp[0] and pedp[0]<vehp[0]+vehl and pedp[1]>L4Y and pedp[1]<L3Y:
                    judge_v=2
    if judge_v==2:
        print('ped-hit')
        crashnumber+=0.5
    return judge_v

def calreward(reward_pre,vehlist,pedlist):
    reward_v=0
    judgeture_v=judgeture(vehlist,pedlist,step)
    if judgeture_v==0:
        reward_v=-0.1
    elif judgeture_v==2:
        reward_v=-1000
    elif judgeture_v==1:
        reward_v=20.0
    return reward_v
       

def force_sum(pedp,vehp,F,P):
    base=(((vehp[0]-pedp[0])**2)+((vehp[1]-pedp[1])**2))**0.5
    Ptemp=[(vehp[0]-pedp[0])*F/base, (vehp[1]-pedp[1])*F/base]
    Pnew=[(P[0]+Ptemp[0]), (P[1]+Ptemp[1])]
    return Pnew

def ped_risk (veha, pedx, pedy, pedv, peda):       
    global L1Y,L2Y,L3Y,L4Y
    ped_incro=0
    ped_outcro=0
    if veha==270 and pedy<L3Y and pedy>L4Y:
        if peda==0:
            ped_incro=(L3Y-pedy)/(pedv+0.1)
        elif peda==180:
            ped_incro=(pedy-L4Y)/(pedv+0.1)
    elif veha==90 and pedy<L1Y and pedy>L2Y:
        if peda==0:
            ped_incro=(L1Y-pedy)/(pedv+0.1)
        elif peda==180:
            ped_incro=(pedy-L2Y)/(pedv+0.1)
    elif veha==90:
        if peda==0 and pedy<L2Y:
            ped_outcro=(L2Y-pedy)/(pedv+0.1)
        elif peda==180 and pedy>L1Y:
            ped_outcro=(pedy-L1Y)/(pedv+0.1)
    elif veha==270:
        if peda==0 and pedy<L4Y:
            ped_outcro=(L4Y-pedy)/(pedv+0.1)
        elif peda==180 and pedy>L3Y:
            ped_outcro=(pedy-L3Y)/(pedv+0.1)
    return (ped_incro, ped_outcro)
   
def calobservation(pedlist, vehlist, observation_type, scandist):
    for veh in vehlist:
        vehp=traci.vehicle.getPosition(veh)
        vehl=traci.vehicle.getLength(veh)
        veha=traci.vehicle.getAngle(veh)
        vehspeed=traci.vehicle.getAngle(veh)
        if veha==270:
            x1=round(L6X,2)
            y1=round(L4Y,2)
            x2=round(L5X,2)
            y2=round(L3Y,2)
        elif veha==90:
            x1=round(L6X,2)
            y1=round(L2Y,2)
            x2=round(L5X,2)
            y2=round(L1Y,2)                      
        obpedlist=[]
        for ped in pedlist:
            pedp=traci.vehicle.getPosition(ped)
            pedspeed=traci.vehicle.getSpeed(ped)
            peda=traci.vehicle.getAngle(ped)
            pedjudge_v=pedjudge(peda,pedp,veha,vehp,vehl)
            
            dist1=(pow((pedp[0]-vehp[0]),2)+pow((pedp[1]-vehp[1]),2))**0.5
            if pedjudge_v==1 and dist1<=scandist:
                obpedlist.append(ped)
        # 对于type1的初始定义
        if observation_type==1:
            P=[0,0]       
        # 对于type2的初始定义
        elif observation_type==2:                
            if obpedlist==[]:
                rect1=8
                rect2=10
                rect3=8
                rect4=10
            else:
                pedp0=traci.vehicle.getPosition(obpedlist[0])
                rect1=pedp0[0]
                rect2=pedp0[1]
                rect3=pedp0[0]
                rect4=pedp0[1]
        # 对于type3的初始定义
        elif observation_type==3:
            out_pedx=8
            out_pedy=10                       
            outcross_ped_rv=1000
            in_pedx=0
            in_pedy=0
            incross_ped_rv=0
        
        for ped in obpedlist:
            pedp=traci.vehicle.getPosition(ped)
            pedspeed=traci.vehicle.getSpeed(ped)
            peda=traci.vehicle.getAngle(ped)
            if observation_type==1:
                # U=10
                # R=0.2
                F=60/dist1
                # F=U*math.exp(-dist1/R)
                P=force_sum(pedp,vehp,F,P)
            if observation_type==2:
                if pedp[0]>rect3:
                    rect3=pedp[0]
                elif pedp[0]<rect1:
                    rect1=pedp[0]
                elif pedp[1]>rect2:
                    rect2=pedp[0]
                elif pedp[1]<rect4:
                    rect4=pedp[0]
            if observation_type==3:                 
                ped_rv=ped_risk (veha, pedp[0], pedp[1], pedspeed, peda)
                if ped_rv[0]==0 and ped_rv[1]<outcross_ped_rv:
                    out_pedx=pedp[0]
                    out_pedy=pedp[1]
                    outcross_ped_rv=ped_rv[1]
                elif ped_rv[1]==0 and ped_rv[0]>incross_ped_rv:
                    in_pedx=pedp[0]
                    in_pedy=pedp[1]   
                    incross_ped_rv=ped_rv[0]        
        if observation_type==1:
            obs_result=[float(P[0]),float(P[1]),float(vehp[0]),float(vehp[1]),float(vehspeed), x1,y1,x2,y2] # 8
        if observation_type==2:
            obs_result=[round(rect1,2),round(rect2,2),round(rect3,2),round(rect4,2),round(vehp[0],2),round(vehp[1],2),round(vehspeed,2),x1,y1,x2,y2] # 10
        if observation_type==3:
            obs_result=[round(out_pedx,2),round(out_pedy,2),round(outcross_ped_rv,2),round(in_pedx,2),round(in_pedy,2),round(incross_ped_rv,2),round(vehp[0],2),round(vehp[1],2),round(vehspeed,2),x1,y1,x2,y2]  # 12
        obs_result=np.array(obs_result)
        return obs_result
            
def GameStart():
    global d
    options=get_options()   
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", "test1v1.sumocfg", "-n", "test1v1.net.xml",
                  "-r", "test1v1.rou.xml"])

    randN=random.randint(0,9)
    d={}
    speedlist=[]
    timelist=[]
    for r in range(sheet1.nrows):
        timelist.append(sheet1.cell_value(r,randN))
    for r in range(sheet2.nrows):
        speedlist.append(sheet2.cell_value(r,randN))  
    #1:C 2:HPS

    # ,Bdist2crosswalk,Bdist2curb)        
    # global p1e, p2e, p1o, p2o, v1o, v1e, vehlf0, vehlf180

    global pedreactiontime,veh_reactiontime,b, scandist, ped_scandist, ped_extratime, ped_type, observation_type
    pedreactiontime=1
    veh_reactiontime=0.1
    # a=2
    b=3
    scandist=50
    ped_scandist=50
    ped_extratime=1
    ped_type=1    
    # observation_type=2

def gamerun(vehlist, pedlist, action,reward_pre,step,observation_type):

    vehicleaction(vehlist, action)   
    observation=calobservation(pedlist, vehlist, observation_type, scandist)
    reward_v=calreward(reward_pre,vehlist,pedlist)
    ture=judgeture(vehlist,pedlist,step)
    datasubscribe_pedm (pedlist)
    datasubscribe_veh (vehlist)
    return observation, reward_v, ture

def datasubscribe_veh (vehlist):
    global step
    for veh in vehlist:
        veha=traci.vehicle.getAngle(veh)
        vehspeed=traci.vehicle.getSpeed(veh)
        vehp=traci.vehicle.getPosition(veh)
        vehl=traci.vehicle.getLength(veh)
        tempdata=np.zeros((1,6))
        tempdata[0][0]=step
        tempdata[0][1]=veha
        tempdata[0][2]=vehp[0]
        tempdata[0][3]=vehp[1]
        tempdata[0][4]=vehspeed
        tempdata[0][5]=vehl
        try:
            globals()['R%s'%veh]
        except:
            globals()['R%s'%veh]=tempdata                
        else:
            globals()['R%s'%veh]=np.vstack([globals()['R%s'%veh],tempdata])
            
def datasubscribe_pedm (pedlist):
    global step
    for ped in pedlist:
        peda=traci.vehicle.getAngle(ped)
        pedspeed=traci.vehicle.getSpeed(ped)
        pedp=traci.vehicle.getPosition(ped)
        tempdata=np.zeros((1,5))
        tempdata[0][0]=step
        tempdata[0][1]=peda
        tempdata[0][2]=pedp[0]
        tempdata[0][3]=pedp[1]
        tempdata[0][4]=pedspeed
        try:
            globals()['R%s'%ped]
        except:
            globals()['R%s'%ped]=tempdata                
        else:
            globals()['R%s'%ped]=np.vstack([globals()['R%s'%ped],tempdata])

def end_episode():
    traci.close()
    for key in list(globals().keys()):
        if key.startswith("p1e"): # 排除系统内建函数
            globals().pop(key)
        elif key.startswith("p1o"): # 排除系统内建函数
            globals().pop(key)
        elif key.startswith("p2e"): # 排除系统内建函数
            globals().pop(key) 
        elif key.startswith("p2o"): # 排除系统内建函数
            globals().pop(key)
        elif key.startswith("v1e"): # 排除系统内建函数
            globals().pop(key)
        elif key.startswith("v1o"): # 排除系统内建函数
            globals().pop(key)
        elif key.startswith("Rperson"): # 排除系统内建函数
            globals().pop(key)
        elif key.startswith("Rflow"): # 排除系统内建函数
            globals().pop(key)            

def listread (pedlist):
    global sheet1, sheet2,d
    for ped in pedlist:
        try:
            globals()['Speed%s'%ped]
        except:
            randN=d[ped]
            timelist=[]
            speedlist=[]
            for r in range(sheet1.nrows):
                timelist.append(sheet1.cell_value(r,randN))
            for r in range(sheet2.nrows):
                speedlist.append(sheet2.cell_value(r,randN))            
            globals()['Time%s'%ped]=timelist
            #exc('print(Time%s)'%ped)
            globals()['Speed%s'%ped]=speedlist

# def nonvehgamerun():
#     traci.simulationStep()
#     vehs = traci.vehicle.getIDList()
#     global pedlist, vehlist
#     group ('person',vehs,pedlist)
#     group ('flow',vehs,vehlist)
#     keycreating (pedlist,2,30,50)
#     listread(pedlist)   
#     pedmovement_One_Stage (pedlist,vehlist,pedreactiontime,step,ped_extratime,ped_type,veh_reactiontime,b,ped_scandist)
#     # pedmovement_Rolling_Gap (pedlist,vehlist,pedreactiontime,step,ped_extratime,ped_type,veh_reactiontime,b,ped_scandist)
#     return vehlist, pedlist

def save1(model,name = 'saved'):
    save_path = './cob3011/'+name
    saver = tf.train.Saver()
    saver.save(model.sess,save_path)
    
def save2(model,name = 'saved'):
    save_path = './cob3012/'+name
    saver = tf.train.Saver()
    saver.save(model.sess,save_path)
    
def save3(model,name = 'saved'):
    save_path = './cob3013/'+name
    saver = tf.train.Saver()
    saver.save(model.sess,save_path)

def save4(model,name = 'saved'):
    save_path = './cob3014/'+name
    saver = tf.train.Saver()
    saver.save(model.sess,save_path)
    
#恢复模型，可以输入name=新文件名防止覆盖
def load(model,name = 'saved'):
    save_path = './cob3010/'+name
    saver = tf.train.Saver()
    saver.restore(model.sess,save_path)
    
def observation_ini(observation_type):
    inputX=0
    inputY=7.2
    crosswidth=4.5
    lanewidth=3.6
    medianwidth=1
    detectLength=3
    waitareaLength=1
    stoplinedistance=2
    # Bdist2crosswalk=2000
    # Bdist2curb=2000
           
    calgeomertry (inputX, inputY, crosswidth, lanewidth, medianwidth, detectLength, waitareaLength, stoplinedistance)
    veha=270
    vehp=[0,0]
    vehspeed=0
    if veha==270:
        x1=round(L6X,2)
        y1=round(L4Y,2)
        x2=round(L5X,2)
        y2=round(L3Y,2)
    elif veha==90:
        x1=round(L6X,2)
        y1=round(L2Y,2)
        x2=round(L5X,2)
        y2=round(L1Y,2)            
    if observation_type==1:
        first_observation=[0,0,float(vehp[0]),float(vehp[1]),float(vehspeed), x1,y1,x2,y2] # 8
    elif observation_type==2:
        first_observation=[8,10,8,10,round(vehp[0],2),round(vehp[1],2),round(vehspeed,2),x1,y1,x2,y2] # 10
    elif observation_type==3:
        first_observation=[round(8,2),round(10,2),round(1000,2),round(8,2),round(10,2),round(0,2),round(vehp[0],2),round(vehp[1],2),round(vehspeed,2),round(x1,2),round(y1,2),round(x2,2),round(y2,2)]  # 12
    first_observation=np.array(first_observation)
    return first_observation

observation_type=3
if observation_type==1:
    observation_space=9
elif observation_type==2:
    observation_space=11
elif observation_type==3:
    observation_space=13
data =xd.open_workbook ('test.xls')
sheet1 = data.sheet_by_name('Sheet1')
sheet2 = data.sheet_by_name('Sheet2')
crashnumber=0
global RL
RL = PolicyGradient(
    n_actions=4,
    n_features=observation_space,
    learning_rate=0.02,
    reward_decay=0.995,
    # output_graph=True,
)
load(RL,name = 'saved')
obvehmodel=1
if obvehmodel==1:
    # lsb=0.8, vehlength=6,vehwidth=2.2
    vehlf0=[-9,-8,-5,-2]
    vehlf180=[-10,-7,-4,-3]
else:
    # lsb=1.3, vehlength=5,vehwidth=2
    vehlf0=[-9,-8,-4,-3]
    vehlf180=[-9,-8,-4,-3]

for i_episode in range(3200):
    pvolume=3
    starttime=round(round(random.uniform(30.00,60.00),0),2)
    RandomSenario(pvolume, starttime)
    # 更新rou文件
    p1e=p2e=p1o=p2o=v1o=v1e={}
    reward_pre=0
    step=0
    judge=0
    observation=observation_ini(observation_type)
    observation=np.array(observation)
    GameStart()
    while judge==0:
        global pedlist, vehlist
        vehlist=[]
        pedlist=[]
        step += 1
        traci.simulationStep()
        vehs = traci.vehicle.getIDList()
        group ('person',vehs,pedlist)
        group ('flow',vehs,vehlist)
        keycreating (pedlist,2,1,30)
        #1:30-->GD 31:54-->LS 55:176-->MS 176:200-->HS
        listread(pedlist)   
        pedmovement_One_Stage (pedlist,vehlist,pedreactiontime,step,ped_extratime,ped_type,veh_reactiontime,b,ped_scandist)
        # pedmovement_Rolling_Gap (pedlist,vehlist,pedreactiontime,step,ped_extratime,ped_type,veh_reactiontime,b,ped_scandist)
        if vehlist!=[]:    
            
            # RL procedure
            action = RL.choose_action(observation)
            # print(action)
            # print(action)
            observation_, reward, judge =  gamerun(vehlist, pedlist, action,reward_pre,step,observation_type)
            # print(observation)
            RL.store_transition(observation, action, reward)
            
            observation = observation_
            # print(observation)
            reward_pre=reward
            
        if judge!=0:
            # calculate running reward
            ep_rs_sum = sum(RL.ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

            print("episode:", i_episode, "  reward:", int(running_reward))
            print("timespend:", round((step/10-starttime),1))
            print("crashrate:", round((crashnumber/(i_episode+1)),1))
            vt = RL.learn()  # train
            if i_episode==799:
                save1(RL,name = 'saved')
            elif i_episode==1599:
                save2(RL,name = 'saved')
            elif i_episode==2399:
                save3(RL,name = 'saved')
            elif i_episode==3199:
                save4(RL,name = 'saved')                
            # # 数据封存
            # if i_episode == 30:
            #     plt.plot(vt)  # plot the episode vt
            #     plt.xlabel('episode steps')
            #     plt.ylabel('normalized state-action value')
            #     plt.show()    
            # break

    end_episode()


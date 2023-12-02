import random
from deap import base
from deap import creator
from deap import tools
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import peakutils
import itertools
import multiprocessing
import sys
from scipy.interpolate import griddata
from scipy.signal import medfilt
from scipy.ndimage import uniform_filter1d
from scipy.stats import entropy, median_abs_deviation

from timeit import default_timer as timer
import datetime
import argparse

def parse_commandline():
    """Parse the arguments given on the command-line.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc",
                        help="Job number",
                        type=int,
                        default=1)
    args = parser.parse_args()
    return args

def median_abs_deviation2(x):
    y = np.array(x)**2
    med = np.median(y, axis = 0)
    return np.median(np.abs(y-med), axis = 0)

(yvals30,levels30,yvals35,levels35,yvals40,levels40,x2,y2,
     ti,ti35,ti0) = pickle.load(open("contour_info30_40.pkl", "rb"), encoding="bytes")

if 1:
    filt2 = [0,2,6,14]
    yBins30 = [yvals30[f2] for f2 in filt2[0:3]] + [yvals30[0]+5]
    yBins35 = [yvals35[f2] for f2 in filt2] + [yvals35[0]+5]
    yBins40 = [yvals40[f2] for f2 in filt2] + [yvals40[0]+5]
else:
    filt2 = [0,6]
    yBins30 = [yvals30[f2] for f2 in filt2] + [yvals30[0]+5]
    yBins35 = [yvals35[f2] for f2 in filt2] + [yvals35[0]+5]
    yBins40 = [yvals40[f2] for f2 in filt2] + [yvals40[0]+5]

yBins = [yBins40, yBins35, yBins30]

xv = np.hstack((np.linspace(-21,-9,endpoint=False), x2,np.linspace(9, 21)))
t30 = np.hstack((np.ones(50)*25, ti[0],np.ones(50)*np.max(ti)))
t35 = np.hstack((np.ones(50)*25, ti35[0],np.ones(50)*np.max(ti35)))
t40 = np.hstack((np.ones(50)*25, ti0[0],np.ones(50)*np.max(ti0)))

def field(x,y,tm,check25):
    if check25:
        return 0.0
    else:
        h1 = (x*y>0)*2 - 1
        d1 = np.min(np.abs(np.array([x,y])), axis=0)
        temp = griddata(xv,tm,d1*h1)
        return (temp-25.)/10.

def get_antL_loc(x,y,theta, BL, AD):
    return x + BL/2*np.cos(theta) - AD/2*np.sin(theta), y + BL/2*np.sin(theta) + AD/2*np.cos(theta)

def get_antR_loc(x,y,theta, BL, AD):
    return x + BL/2*np.cos(theta) + AD/2*np.sin(theta), y + BL/2*np.sin(theta) - AD/2*np.cos(theta)

def get_head_loc(x,y,theta, BL):
    return x + BL/2*np.cos(theta), y + BL/2*np.sin(theta)

def ou(tau, sig, t, y0=0):
    dt=t[1]-t[0]
    sqrtdt = np.sqrt(dt)
    y=np.zeros(t.shape)
    y[0]=y0
    wt=sqrtdt*np.random.normal(size=t.shape)
    for i in range(len(t)-1):
            y[i+1]=y[i] + (-y[i]*dt + sig*wt[i])/tau
    return y

def h(s,a,b):
    return 1./(1+np.exp(-a*s+b))

def reflect(x,y,theta,wallRadius):
    # find intersection of body axis vector and circle
    vx,vy = np.cos(theta),np.sin(theta)
    a = (vx**2+vy**2)
    b = 2.*(x*vx+y*vy) 
    c = x**2 + y**2 - wallRadius**2.
    if b**2 -4.*a*c <0:
        return theta+np.pi
    tArray = [(-b +np.sqrt(b**2 -4.*a*c))/(2.*a),(-b -np.sqrt(b**2 -4.*a*c))/(2.*a)]

    i1 = np.argmin(np.abs(tArray))
    pos = np.array([x,y]) + tArray[i1]*np.array([vx,vy])
    
    ## perform reflection
    ang2 = np.arctan2(pos[1],pos[0])
    ang1 = np.arctan2(vy,vx)
    l1  = np.argmin(np.abs([ang1-ang2,ang1-ang2-2*np.pi,ang1-ang2+2*np.pi]))
    l0 = [ang1-ang2,ang1-ang2-2*np.pi,ang1-ang2+2*np.pi]
    l0 = l0[l1]

    thetaNew = theta +2* np.sign(l0)*(np.pi/2.-np.abs(l0))#+random.gauss(0,np.pi/10)

    return thetaNew

def h2(s,a,b,mult,deriv):
    return 1./(1+np.exp(-a*s - mult*deriv + b))

def simulate(weights, fixed, tm, check25, T, div):
    # fixed things
    wI = 40*weights[0]
    wC = 40*weights[1]
    a = 10*weights[2]
    b = 10*weights[3]
    tau_m = weights[4]
    sig_m = weights[5]
    tau_s = weights[6]
    sig_s = weights[7]/100
    kd30 = weights[8]
    kd35 = weights[9]
    kd40 = weights[10]

    tp = weights[11]

    BL = fixed[0]
    AD = fixed[1]
    v0 = fixed[2]
    d = fixed[3]
    wallRadius = fixed[4]
    #tau_s = fixed[5]
    #sig_s = fixed[6]

    # simulate
    badFlag = False
    t = np.linspace(0, T, T*div, endpoint=False)
    dt = t[1] - t[0]
    epsL = ou(tau_s, sig_s, t)
    epsR = ou(tau_s, sig_s, t)
    gam = ou(tau_m, sig_m, t)
    sol = np.zeros((len(t),3))
    sol[0,:] = np.array([8, (np.random.randint(2)*2-1)*8, 2*np.pi*np.random.rand()])

    pL, pR = np.ones(t.shape), np.ones(t.shape)
    deduct = h(-0.1,a,b)
    for i in range(len(t)-1):
        x,y,theta = sol[i,:]
        xyLA = get_antL_loc(x,y,theta, BL, AD)
        xyRA = get_antR_loc(x,y,theta, BL, AD)

        if np.linalg.norm(xyLA)>wallRadius or np.linalg.norm(xyRA)>wallRadius:
            theta = reflect(x,y,theta,wallRadius)
            xyLA = get_antL_loc(x,y,theta, BL, AD)
            xyRA = get_antR_loc(x,y,theta, BL, AD)
        
        sL = field(xyLA[0],xyLA[1],tm,check25) + epsL[i]
        sR = field(xyRA[0],xyRA[1],tm,check25) + epsR[i]
        yL, yR = h(sL,a,b)-deduct, h(sR,a,b)-deduct
        
        if i == 0:
            pL[0] = 1/yL
            pR[0] = 1/yR
        
        # relu or abs
        dL = max(pL[i]*yL - 1,0)
        dR = max(pR[i]*yR - 1,0)
        pL[i+1] = pL[i] + dt*(-pL[i] + 1/(yL))/tp
        pR[i+1] = pR[i] + dt*(-pR[i] + 1/(yR))/tp

        if tm[-1] == t40[-1]:
            kd = kd40
        elif tm[-1] == t35[-1]:
            kd = kd35
        elif tm[-1] == t30[-1]:
            kd = kd30
        elif check25 == 1:
            kd = 0
        
        vL = wI*(h(sL,a,b) + kd*dL) + wC*(h(sR,a,b) + kd*dR) + v0 + gam[i]
        vR = wC*(h(sL,a,b) + kd*dL) + wI*(h(sR,a,b) + kd*dR) + v0 - gam[i] 

        fx = [0.5*(vL+vR)*np.cos(theta), 0.5*(vL+vR)*np.sin(theta), (vR-vL)/d]

        if np.linalg.norm(fx[:2]) > 25:
            badFlag = True

        sol[i+1,:] = np.array([x,y,theta]) + dt*np.array(fx)

    return sol, badFlag

def avoidance_calc(trajectory):
    trajectory = trajectory[:,0:2]
    hotFrac = np.sum(np.prod(trajectory,axis=1)>0)*1.0/trajectory.shape[0]
    return 1.0-2.*hotFrac

def get_dists(trajectory):
    return np.min(np.abs(trajectory),axis=1)*(2.*(np.prod(trajectory,axis=1)>0)-1)

def get_quads(trajectory):
    quads = []
    for i in range(trajectory.shape[0]):
        if trajectory[i,0]>0 and trajectory[i,1]>0:
            quads.append(1) 
        elif trajectory[i,0]<0 and trajectory[i,1]>0:
            quads.append(2) 
        elif trajectory[i,0]<0 and trajectory[i,1]<0:
            quads.append(3)
        else:
            quads.append(4) 
    return quads

def cross_turn_ratio(trajectory, threshMax, BL):
    # boundary interaction starts at 25.5C as before
    crosses,turns = 0,0
    head_locs = np.array(get_head_loc(trajectory[:,0], trajectory[:,1], trajectory[:,2], BL)).T
    head_dists = get_dists(head_locs)
    head_quads = get_quads(head_locs)
    inBehavior = False
    for i in range(trajectory.shape[0]):
        if not inBehavior and head_dists[i]>threshMax and head_dists[i]<(threshMax+5.) and head_dists[i-1]<threshMax:
            inBehavior = True
            startQuad = head_quads[i]
        elif inBehavior and head_dists[i]>threshMax and head_dists[i]<(threshMax+5.):
            pass
        elif inBehavior and head_dists[i]>threshMax and head_dists[i]>(threshMax+5.)and head_dists[i-1]<(threshMax+5.):
            inBehavior=False
            if (startQuad==2 and head_quads[i]!=4) or  (startQuad==4 and head_quads[i]!=2):
                crosses+=1
        elif inBehavior and head_dists[i]<threshMax and head_dists[i]<(threshMax+5.) and head_dists[i-1]>threshMax:
            inBehavior=False
            if (startQuad==2 and head_quads[i]!=4) or  (startQuad==4 and head_quads[i]!=2):
                turns+=1
    if crosses+turns>0:
        return turns/(crosses+turns),turns,crosses
    else:
        return np.nan,turns,crosses
    # if we leave the hot region before getting to +5cm (by crossing at the middle point), disregard the interaction

def initAngle(quad,state):
    vec = np.array([np.cos(state[2]),np.sin((state[2]))])
    if quad==2:
        if abs(state[1])<abs(state[0]):
            return np.arccos(np.dot(vec,[-1,0]))
        else:
            return np.arccos(np.dot(vec,[0,-1]))
    elif quad==4:
        if abs(state[1])<abs(state[0]):
            return np.arccos(np.dot(vec,[1,0]))
        else:
            return np.arccos(np.dot(vec,[0,1]))

def movingaverage(interval, window_size):
    return uniform_filter1d(interval, size=window_size, mode='nearest')

def input_output_calc(trajectory, dt, tm, threshMax, BL, AD, wallRadius):
    # boundary interaction starts at 25.5C as before
    head_locs = np.array(get_head_loc(trajectory[:,0], trajectory[:,1], trajectory[:,2], BL)).T
    head_dists = get_dists(head_locs)
    head_quads = get_quads(head_locs)
    inBehavior = False
    approachAngles,turnAngles = [],[]
    angVels = movingaverage(medfilt(np.gradient(trajectory[:,2])/dt/30.,9),3)
    antDiff = []
    for i in range(trajectory.shape[0]):
        if not inBehavior and head_dists[i]>threshMax and head_dists[i]<(threshMax+5.) and head_dists[i-1]<threshMax:
            inBehavior = True
            startQuad = head_quads[i]
            approachAngle = initAngle(startQuad,trajectory[i,:])
            startOrientation = trajectory[i,2]
            startFrame = i-1

        elif inBehavior and head_dists[i]>threshMax and head_dists[i]<(threshMax+5.):
            pass
        elif inBehavior and head_dists[i]>threshMax and head_dists[i]>(threshMax+5.)and head_dists[i-1]<(threshMax+5.):
            inBehavior=False
            if (startQuad==2 and head_quads[i]!=4) or  (startQuad==4 and head_quads[i]!=2) and np.all(np.linalg.norm(head_locs[startFrame:i],axis=1)<(wallRadius-2)):
                # approachAngles.append(approachAngle)
                turnStart = get_turn_start(angVels[startFrame:i]) +startFrame
                if turnStart!=startFrame or 1:
                    approachAngles.append(initAngle(startQuad,[head_locs[turnStart,0],head_locs[turnStart,1],trajectory[turnStart,2]]))
                    turnEnd = get_turn_end(angVels[turnStart:i],turnStart)+startFrame
                    turnAngles.append(trajectory[turnEnd,2]- trajectory[turnStart,2])
                    ant_locs = np.array([get_antL_loc(trajectory[turnStart,0],trajectory[turnStart,1],trajectory[turnStart,2], BL, AD), get_antR_loc(trajectory[turnStart,0],trajectory[turnStart,1],trajectory[turnStart,2], BL, AD)])
                    ant_temps = np.array([field(ant_locs[0][0],ant_locs[0][1],tm,0),field(ant_locs[1][0],ant_locs[1][1],tm,0)])
                    antDiff.append(10*(ant_temps[1]-ant_temps[0]))
        elif inBehavior and head_dists[i]<threshMax and head_dists[i]<(threshMax+5.) and head_dists[i-1]>threshMax:
            inBehavior=False
            if (startQuad==2 and head_quads[i]!=4) or  (startQuad==4 and head_quads[i]!=2)and np.all(np.linalg.norm(head_locs[startFrame:i],axis=1)<(wallRadius-2)):
                # approachAngles.append(approachAngle)
                # turnAngles.append(trajectory[i,2]- startOrientation)
                turnStart = get_turn_start(angVels[startFrame:i]) +startFrame
                if turnStart!=startFrame or 1:
                    approachAngles.append(initAngle(startQuad,[head_locs[turnStart,0],head_locs[turnStart,1],trajectory[turnStart,2]]))
                    turnEnd = get_turn_end(angVels[startFrame:i],turnStart-startFrame)+startFrame

                    # plt.plot(180./np.pi*angVels[startFrame:i])
                    # plt.scatter(turnStart-startFrame,180./np.pi*angVels[turnStart],color='g')
                    # plt.scatter(turnEnd-startFrame,180./np.pi*angVels[turnEnd],color='r')
                    # plt.show()
                    turnAngles.append(trajectory[turnEnd,2]- trajectory[turnStart,2])
                    ant_locs = np.array([get_antL_loc(trajectory[turnStart,0],trajectory[turnStart,1],trajectory[turnStart,2], BL, AD), get_antR_loc(trajectory[turnStart,0],trajectory[turnStart,1],trajectory[turnStart,2], BL, AD)])
                    ant_temps = np.array([field(ant_locs[0][0],ant_locs[0][1],tm,0),field(ant_locs[1][0],ant_locs[1][1],tm,0)])
                    antDiff.append(10*(ant_temps[1]-ant_temps[0]))

    return approachAngles,turnAngles,antDiff

def get_turn_start(thetaTraj):
    #define turn threshold speed.
    rotThresh = 1.5
    thetaTraj = thetaTraj*180./np.pi
    tStart = np.nan	
    #get first turn starting in region. 
    #if turn had already started and no other turn happens after, take first frame. 
    for j11 in range(1,len(thetaTraj)):
        if abs(thetaTraj[j11]) >rotThresh and abs(thetaTraj[j11-1]) <rotThresh:
            tStart = j11
            break
    if np.isnan(tStart):
        tStart = np.argmax(np.abs(thetaTraj))
    #now step backwards in time to find the "true" turn start
    # slopeSign = np.sign(thetaTraj[tStart]-thetaTraj[tStart-1])
    for j11 in reversed(range(0,tStart)):
        if np.sign(thetaTraj[j11])!=np.sign(thetaTraj[tStart]):
            turnStart = j11+1
            return turnStart
        # elif np.sign(thetaTraj[j11+1]-thetaTraj[j11])!=slopeSign# and np.sign(thetaTraj[j11+1]-thetaTraj[j11])!=0:
        # 	turnStart = j11+1
        # 	return turnStart
    # print('No turn initiated in region for behavior. Using first frame. ')
    return 0

def get_turn_end(thetaTraj,turnStart):
    rotThresh = 1.5
    thetaTraj = thetaTraj*180./np.pi
    tend = np.nan	
    #find time of passing below threshold
    for j11 in range(turnStart,len(thetaTraj)):
        if abs(thetaTraj[j11]) <rotThresh or np.sign(thetaTraj[j11])!=np.sign(thetaTraj[turnStart]):
            tend = j11
            break
    if np.isnan(tend)or tend==(len(thetaTraj)-1):
        return len(thetaTraj)-1
    #now step forwards in time to find the "true" turn end
    # slopeSign = np.sign(thetaTraj[tend+1]-thetaTraj[tend])
    for j11 in range(tend,len(thetaTraj)-1):
        if np.sign(thetaTraj[j11])!=np.sign(thetaTraj[turnStart]):
            turnEnd = j11-1
            return turnEnd
        # elif np.sign(thetaTraj[j11+1]-thetaTraj[j11])!=slopeSign:
        # 	turnEnd = j11
        # 	return turnEnd
    # print('No turn initiated in region for behavior. Using first frame. ')
    return len(thetaTraj)-1

def rotate_traj(a11,theta1):
    a11 = np.array(a11)
    rotMat = np.array([[np.cos(theta1),-np.sin(theta1)],[np.sin(theta1),np.cos(theta1)]])
    for j1 in range(0,a11.shape[0]):
        a11[j1,:] = np.dot(rotMat,np.squeeze(a11[j1,:]))
    return a11

def rotAngleCalc(coords,q1):
    if abs(coords[0])<abs(coords[1]):
        ax1='vert'
    else:
        ax1='horiz'
    if (q1 ==1 and ax1=='horiz'):
        return 0 
    elif q1 ==1 and ax1 =='vert':
        return 90
    elif (q1 ==2 and ax1=='horiz'):
        return 0
    elif q1 ==2 and ax1 =='vert':
        return -90
    elif q1 ==3 and ax1 =='horiz':
        return -180
    elif q1 ==3 and ax1 =='vert':
        return -90
    elif q1 ==4 and ax1 =='vert':
        return 90
    elif q1 ==4 and ax1 =='horiz':
        return -180

def staff_trajs(trajectory, BL, threshMax, dt, wallRadius):
    # boundary interaction starts at 25.5C as before
    crosses,turns = 0,0
    head_locs = np.array(get_head_loc(trajectory[:,0], trajectory[:,1], trajectory[:,2], BL)).T
    centroid_locs = np.array(trajectory[:,0:2])
    head_dists = get_dists(head_locs)
    head_quads = get_quads(head_locs)
    inBehavior = False
    approachAngles,turnAngles = [],[]
    trajs = []
    vels = []
    cOrT = []
    maxRotData = []
    for i in range(trajectory.shape[0]):
        if not inBehavior and head_dists[i]>threshMax and head_dists[i]<(threshMax+5.) and head_dists[i-1]<threshMax:
            inBehavior = True
            startQuad = head_quads[i]
            approachAngle = initAngle(startQuad,trajectory[i,:])
            theta0 = rotAngleCalc(head_locs[i],head_quads[i])*np.pi/180.
            startOrientation = trajectory[i,2]
            startFrame = i-2
        elif inBehavior and head_dists[i]>threshMax and head_dists[i]<(threshMax+5.):
            pass
        elif inBehavior and head_dists[i]>threshMax and head_dists[i]>(threshMax+5.)and head_dists[i-1]<(threshMax+5.):
            inBehavior=False
            if (startQuad==2 and head_quads[i]!=4) or  (startQuad==4 and head_quads[i]!=2):
                crosses+=1
                trajs.append(rotate_traj(head_locs[startFrame:i+2,:],theta0))
                vels.append(np.linalg.norm(np.gradient(centroid_locs[startFrame:i+2,:],axis=0),axis=1)/dt)
                # vels.append(movingaverage(medfilt(np.sqrt(np.gradient(trajectory[startFrame:i+2,0])**2+np.gradient(trajectory[startFrame:i+2,1])**2)/dt,9),3))
                cOrT.append('c')
                #calculate max turn vel, translational speed at that point
                t11 = movingaverage(medfilt(np.sqrt(np.gradient(trajectory[startFrame:i+2,0])**2+np.gradient(trajectory[startFrame:i+2,1])**2)/dt,5),3)
                a11 = movingaverage(medfilt(np.gradient(trajectory[startFrame:i+2,2])/dt*180./np.pi,5),3)
                maxRotInd = np.argmax(np.abs(a11))
                maxRotData.append((t11[maxRotInd],a11[maxRotInd],np.all(np.linalg.norm(head_locs[startFrame:i],axis=1)<(wallRadius-3))))
        elif inBehavior and head_dists[i]<threshMax and head_dists[i]<(threshMax+5.) and head_dists[i-1]>threshMax:
            inBehavior=False
            if (startQuad==2 and head_quads[i]!=4) or  (startQuad==4 and head_quads[i]!=2):
                turns+=1
                trajs.append(rotate_traj(head_locs[startFrame:i+2,:],theta0))
                # vels.append(movingaverage(medfilt(np.sqrt(np.gradient(trajectory[startFrame:i+2,0])**2+np.gradient(trajectory[startFrame:i+2,1])**2)/dt,9),3))
                vels.append(np.linalg.norm(np.gradient(centroid_locs[startFrame:i+2,:],axis=0),axis=1)/dt)
                cOrT.append('t')
                t11 = movingaverage(medfilt(np.sqrt(np.gradient(trajectory[startFrame:i+2,0])**2+np.gradient(trajectory[startFrame:i+2,1])**2)/dt,5),3)
                a11 = movingaverage(medfilt(np.gradient(trajectory[startFrame:i+2,2])/dt*180./np.pi,5),3)
                maxRotInd = np.argmax(np.abs(a11))
                maxRotData.append((t11[maxRotInd],a11[maxRotInd],np.all(np.linalg.norm(head_locs[startFrame:i],axis=1)<(wallRadius-3))))
    return trajs,vels,cOrT,maxRotData


def getDistance_NumTurns(trajectory1, dt, wallRadius,BL):
    trajectory = np.array(trajectory1)
    winSize = int(np.round(9./30./dt))
    trajectory[:,0] = movingaverage(trajectory[:,0],winSize)
    trajectory[:,1] = movingaverage(trajectory[:,1],winSize)
    trajectory[:,2] = movingaverage(trajectory[:,2],winSize)
    t11 = movingaverage(np.sqrt(np.gradient(trajectory[:,0])**2+np.gradient(trajectory[:,1])**2),winSize)
    a11 = movingaverage(np.gradient(trajectory[:,2])*180./np.pi,winSize)

    indexes = list(peakutils.indexes(a11, thres=0.4, min_dist=int(np.round(6./30./dt))))

    indexes = [i0 for i0 in indexes if a11[i0]>1.5*30*dt and np.linalg.norm(trajectory[i0,0:2])<(wallRadius-BL)]
    indexes2 = list(peakutils.indexes(-a11, thres=0.4, min_dist=int(np.round(6./30./dt))))
    indexes2 = [i0 for i0 in indexes2 if a11[i0]<-1.5*30.*dt and np.linalg.norm(trajectory[i0,0:2])<(wallRadius-BL)]
    indexes.extend(indexes2)
    indexes = np.unique(indexes)

    return np.sum([t11[i1] for i1 in range(0,len(t11)) if np.linalg.norm(trajectory[i1,0:2])<(wallRadius-BL)]), len(indexes)

# DEAP

def unique_check(list11): 
   checkList = []
   for c in list11:
       if c not in checkList:
           checkList.append(c)
   return checkList

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def prior():
    #inputVals = np.array([0.725, -0.56, 0.5044506914904836, 0.3887560820695657, 0.6485170151559537, 0.3907077496945611])
    return list((np.random.rand(1))) + list((-np.random.rand(1))) + list((np.random.rand(9))) + list(0.05+0.25*(np.random.rand(1)))

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')
toolbox.register("individual", tools.initIterate,  creator.Individual, prior)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# the goal ('fitness') function to be maximized
def evaluate(inputVals):
    opt1 = 0.2
    opt2 = np.array([0.95, 0.9, 0.75])
    opt3 = np.array([0.9, 0.8, 0.55])
    opt4 = np.array([0.65, 0.7, 0.9, 0.92, 0.93, 0.94])
    opt5 = np.array([0.49, 0.27, 0.12, 0.12, 0.47, 0.23, 0.22, 0.08 , 0.375, 0.375, 0.25])
    opt6 = np.array([0.12, 0.34, 0.66, 0.88])

    numRuns = 50
    div = 30
    T = 60
    dt = 1./div

    BL = 3.
    AD = 0.3
    v0 = 5
    d = 0.75
    wallRadius = 19
    tau_s = 0.7509151388821899
    sig_s = -0.006675258791037683
    fixed = [BL, AD, v0, d, wallRadius, tau_s, sig_s]
    TM = [t40, t35, t30]

    badFlag = False

    # stats to store
    turns = [[],[],[]]
    crosses = [[],[],[]]
    avoidances = [[],[],[]]
    inAnglesList = [[],[],[]]
    outAnglesList = [[],[],[]]
    inDiffsList = [[],[],[]]
    trajsList = [[],[],[]]
    maxTsTurns = [[],[],[]]
    staffRatio = [None,None,None]

    badReturn = (500, 4.0, 2.0, 2.0, 2.0)

    for kk in range(3):
        tm = TM[kk]
        threshMax = griddata(tm,xv,25.5)
        tMax2 = griddata(tm,xv,26.5)
        for j in range(numRuns):
            if badFlag:
                return badReturn
            ySol, badFlag = simulate(inputVals, fixed, tm, 0, T, div)
            c1 = cross_turn_ratio(ySol,threshMax,BL)
            '''
            while np.isnan(c1[0]):
                ySol = simulate(inputVals, fixed, tm, 0, T, div)
                c1= cross_turn_ratio(ySol,threshMax,BL)
            '''
            turns[kk].append(c1[1])
            crosses[kk].append(c1[2])
            avoidances[kk].append(avoidance_calc(ySol))

            inAngles,outAngles,inDiffs = input_output_calc(ySol, dt, tm, threshMax, BL, AD, wallRadius)
            inAnglesList[kk].append(inAngles)
            outAnglesList[kk].append(outAngles)
            inDiffsList[kk].append(inDiffs)

            traj,vel,cOrT,maxRots = staff_trajs(ySol, BL, threshMax, dt, wallRadius)
            trajsList[kk].extend(traj)

        inAnglesList[kk] = np.concatenate(inAnglesList[kk])
        outAnglesList[kk] = np.concatenate(outAnglesList[kk])
        inDiffsList[kk] = np.concatenate(inDiffsList[kk])
        maxTsTurns[kk] = -1.*np.array([np.min(trajsList[kk][i1][:,1]) for i1 in range(len(trajsList[kk])) if np.min(trajsList[kk][i1][:,1])>-1.*(threshMax+5)])
        cts, _ = np.histogram(maxTsTurns[kk], bins = yBins[kk])
        if len(maxTsTurns[kk]) > 0.2: # if not zero
            staffRatio[kk] = cts/len(maxTsTurns[kk])
        else:
            staffRatio[kk] = 10*np.ones(cts.size)

    distTurns = []
    for j in range(numRuns):
        ySol, badFlag = simulate(inputVals, fixed, tm, 1, T, div)
        if badFlag:
            return badReturn
        c1 = cross_turn_ratio(ySol,threshMax,BL)
        '''
        while np.isnan(c1[0]):
            ySol = simulate(inputVals, fixed, tm, 1, T, div)
            c1 = cross_turn_ratio(ySol,threshMax,BL)
        '''
        distTurns.append(getDistance_NumTurns(ySol, dt, wallRadius, BL))
        
    tPD1 = np.sum(distTurns,axis=0)
    errorTurnsPerDist = np.nan_to_num(tPD1[1]/tPD1[0], nan = 500)
    
    av_index = np.array([np.median(avoidances[0]),np.median(avoidances[1]),np.median(avoidances[2])])
    
    ratios = []
    for kk in range(3):
        tc = np.sum(turns[kk])+np.sum(crosses[kk])
        if tc > 0.2: # if not zero
            ratios.append(np.sum(turns[kk])/tc)
        else:
            ratios.append(10)
            #badFlag = True
    ratios = np.array(ratios)
    
    tDiff = np.concatenate(inDiffsList)
    outThetas =  np.array(np.concatenate(outAnglesList))*180./np.pi

    ###for predictive analysis of turns based on temp difference
    lDiffs = [np.abs(tDiff[i]) for i in range(0,len(tDiff))if tDiff[i]>0. and abs(outThetas[i])>15.]
    rDiffs = [np.abs(tDiff[i]) for i in range(0,len(tDiff))if tDiff[i]<0. and abs(outThetas[i])>15.]
    lOuts = [outThetas[i]>0. for i in range(0,len(tDiff))if tDiff[i]>0. and abs(outThetas[i])>15.]
    rOuts = [outThetas[i]<0. for i in range(0,len(tDiff))if tDiff[i]<0. and abs(outThetas[i])>15.]
    tdiff1 = lDiffs+rDiffs
    tOuts1 = lOuts+rOuts
    bins = np.linspace(0,.5,6)
    inds1 = np.digitize(tdiff1,bins)-1
    tdiffratios=[]
    for i in range(0,len(bins)):
        tdiffratios.append(np.mean([tOuts1[j]*1.0 for j in range(0,len(inds1)) if inds1[j]==i]))
    tdiffratios = np.array(tdiffratios)

    ### for polar plots

    inThetas = inAnglesList[0]*180./np.pi
    outThetas = outAnglesList[0]*180./np.pi
    bins = np.linspace(0,180,5)
    halfPts = (bins[1]-bins[0])/2 + bins[0:len(bins)-1]
    inds = np.digitize(inThetas,bins,right=True)
    # print(inds,bins)
    inds = inds-1
    numSlots = len(bins)-1
    rCount = np.zeros(numSlots)
    lCount = np.zeros(numSlots)
    sCount = np.zeros(numSlots)
    angThres = 15
    for i in range(0,len(inThetas)):
        if outThetas[i]>angThres:
            lCount[inds[i]]+=1

        elif outThetas[i]<-angThres:
            rCount[inds[i]]+=1

        else:
            sCount[inds[i]]+=1

    rCountNorm = np.zeros(len(rCount))
    sCountNorm = np.zeros(len(rCount))
    lCountNorm = np.zeros(len(rCount))
    for i in range(0,len(rCount)):
        tot = rCount[i]+lCount[i]#+sCount[i]
        if tot > 0.2: # check not zero
            rCountNorm[i] = rCount[i]/tot
            # sCountNorm[i]= sCount[i]/tot
            lCountNorm[i] = lCount[i]/tot
        else:
            rCountNorm[i] = 10
            lCountNorm[i] = 10
    '''
    # return calculations
    opt2p = (1-opt2)/2
    av_indexp = (1-av_index)/2
    staffp = np.concatenate(staffRatio)
    opt2r = np.hstack((opt2p, 1-opt2p))
    opt3r = np.hstack((opt3, 1-opt3))
    opt6r = np.hstack((opt6, 1-opt6))
    opt5r = np.hstack((opt5, 1-opt5))

    returnal = (np.linalg.norm(errorTurnsPerDist-opt1), entropy(opt2r, qk = np.hstack((av_indexp, 1-av_indexp))), \
        entropy(opt3r, qk = np.hstack((ratios, 1-ratios))), entropy(opt6r, qk = np.hstack((lCountNorm, 1-lCountNorm))), entropy(opt5r, qk = np.hstack((staffp, 1-staffp))) )
    
    for loss in returnal:
        if loss > 10:
            badFlag = True

    returnal = [np.linalg.norm(errorTurnsPerDist-opt1), np.linalg.norm(av_index-opt2), \
        np.linalg.norm(ratios-opt3), np.linalg.norm(lCountNorm-opt6), np.linalg.norm(np.concatenate(staffRatio)-opt5)]
    '''

    idx_staff = [0,4,8]
    staff_Bin1 = np.array([z[0] for z in staffRatio])

    returnal = [np.linalg.norm(errorTurnsPerDist-opt1), np.linalg.norm(av_index-opt2), \
        np.linalg.norm(ratios-opt3), np.linalg.norm(lCountNorm-opt6), np.linalg.norm(staff_Bin1-opt5[idx_staff])]

    for i in range(len(returnal)):
        if returnal[i] > badReturn[i]:
            returnal[i] = badReturn[i]
    
    if badFlag:
        return badReturn
    else:
        return returnal
#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", evaluate)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.25, indpb=0.25)

def checkBounds():
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in [0, 2, 8, 9, 10]:
                    if child[i]< 0:
                        child[i] = 0
                for i in [1]:
                    if child[i] > 0:
                        child[i] = 0
                for i in [4, 6]:
                    if child[i] < 0.1:
                        child[i] = 0.1

                if child[11] < 0.05:
                    child[11] = 0.05
                if child[11] > 0.3:
                    child[11] = 0.3
            return offspring
        return wrapper
    return decorator

toolbox.decorate("mutate", checkBounds())
toolbox.decorate("mate", checkBounds())

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selNSGA2_spdr)

def main(tot_pop, num_gens, checkpoint):
    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    if checkpoint:
        with open('checkpoint.pkl', "rb") as cp_file:
            cp = pickle.load(cp_file)
        logbook = cp['logbook']
        pop = cp['population']
        gen = cp['generation']
        pareto = cp["pareto"]
        hof = cp["hof"]
        random.setstate(cp["rndstate"])
        print("Continuing evolution from generation", gen)
    else:
        pop = toolbox.population(n=tot_pop)
        gen = 0
        pareto = tools.ParetoFront()
        logbook = tools.Logbook()
        hof = tools.HallOfFame(2*tot_pop*num_gens)
        initial_state = random.getstate()
        print("Start of evolution")


    # statistics
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("med", np.median, axis = 0)
    stats.register("med_std", median_abs_deviation, axis=0)
    stats.register("med_std_sq", median_abs_deviation2)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    stats2 = tools.Statistics(key=lambda ind: ind)
    stats2.register("avg2", np.mean, axis=0)
    stats2.register("std2", np.std, axis=0)
    stats2.register("med2", np.median, axis = 0)
    stats2.register("med_std2", median_abs_deviation, axis=0)
    stats2.register("min2", np.min, axis=0)
    stats2.register("max2", np.max, axis=0)

    # CXPB  is the probability with which two individuals are crossed
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.5

    # Evaluate the entire population
    jobs = toolbox.map(toolbox.evaluate, pop)
    fitnesses = jobs.get()
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, tot_pop)
    
    print("  Evaluated %i individuals" % len(pop))


    # Variable keeping track of the number of generations
    FREQ = 10

    # Begin the evolution
    while gen < num_gens:
        # record statistics
        record = stats.compile(pop)
        record2 = stats2.compile(pop)
        logbook.record(gen=gen, **record)
        logbook.record(gen=gen, **record2)

        # A new generation
        gen = gen + 1
        print("-- Generation %i --" % gen)

        # Select the next generation individuals
        offspring = tools.selTournamentDCD(unique_check(pop), tot_pop)
        
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        jobs = toolbox.map(toolbox.evaluate, invalid_ind)
        fitnesses = jobs.get()
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pareto.update(pop+offspring)
        hof.update(pop+offspring)
        pop = toolbox.select(unique_check(pop+offspring),tot_pop)

        if gen % FREQ == 0:
        # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=pop, generation=gen, rndstate=random.getstate(), pareto = pareto, hof = hof, logbook = logbook, init_state = initial_state)
            with open("checkpoint.pkl", "wb") as cp_file:
                pickle.dump(cp, cp_file)

    print("-- End of (successful) evolution --")

    # Store best ones at the end
    notes = "Jul 8, 2023: (aushra, optimize tp, try2) param stats, optimize bin1, adjust kd across 3, relu, one sided sum"
    cp = dict(population=pop, generation=gen, rndstate=random.getstate(), pareto = pareto, hof = hof, logbook = logbook, init_state = initial_state, message = notes)
    with open("checkpoint.pkl", "wb") as cp_file:
        pickle.dump(cp, cp_file)

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

if __name__ == '__main__':
    args = parse_commandline()
    print("Number of cpus Python thinks you have : ", multiprocessing.cpu_count())
    pool = multiprocessing.Pool(args.nproc)

    toolbox.register('map', pool.map_async)
    tic = timer()
    main(tot_pop=112, num_gens=500, checkpoint=False)
    pool.close()
    print(str(datetime.timedelta(seconds=int(timer()-tic))))
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 18:00:45 2021

@author: guila
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 13:56:52 2021

@author: guila
"""

from scipy.integrate import odeint
import numpy as np


def vf_mixed(x,t, k1, r_obstacle, k2, rotation, r, po, pc):
    
    dxo = np.zeros(2)
    norm = np.sqrt((x[0]-po[0])**2+(x[1]-po[1])**2)
    if norm < r_obstacle : 
        dxo[0] = np.sign(x[0]-po[0])*0.5*k1*((1/norm)-(1/r_obstacle))**2
        dxo[1] = np.sign(x[1]-po[1])*0.5*k1*((1/norm)-(1/r_obstacle))**2
        
    else:
        dxo[0] = 0
        dxo[1] = 0
    
    xc = np.zeros(2)
    xc[0] = np.sqrt((x[0]-pc[0])**2+(x[1]-pc[1])**2)
    xc[1] = np.nan_to_num(np.arctan2(x[1]-pc[1],x[0]-pc[0]))
    dx = vf_circle(xc, 0, k2, rotation, r)
    
    dxc = np.zeros(2)
    dxc[0] = dx[0]*np.cos(xc[1])+xc[0]*dx[1]*-np.sin(xc[1])
    dxc[1] = dx[0]*np.sin(xc[1])+xc[0]*dx[1]*np.cos(xc[1])
    
    return dxc[0]+dxo[0], dxc[1]+dxo[1]

def vector_field_mixed(xlim, ylim, k1, r_obstacle, k2, rotation, r, po, pc):
    
    grid_x = round((xlim[1]-xlim[0])*2.5)
    grid_y = round((ylim[1]-ylim[0])*2.5)
    X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], grid_x), np.linspace(ylim[0], ylim[1], grid_y))
    
    U=np.zeros(X.shape)
    V=np.zeros(Y.shape)
    
    Rc = np.sqrt((X-pc[0])**2+(Y-pc[1])**2)
    Tc = np.nan_to_num(np.arctan2(Y-pc[1],X-pc[0]))
    Uc=np.zeros(Rc.shape)
    
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            norm = np.sqrt((X[i,j]-po[0])**2+(Y[i,j]-po[1])**2)
            if norm < r_obstacle : 
                U[i,j] = np.sign(X[i,j]-po[0])*0.5*k1*((1/norm)-(1/r_obstacle))**2
                V[i,j] = np.sign(Y[i,j]-po[1])*0.5*k1*((1/norm)-(1/r_obstacle))**2
            
            else:
                U[i,j] = 0
                V[i,j] = 0
            
    No = np.sqrt(U** 2 + V** 2)
    Uo = np.nan_to_num(U / No)
    Vo = np.nan_to_num(V/ No)
    
    if rotation:
        Uc = (r-Rc)*k2
        Vc = np.ones(Y.shape)
    
    else:
        Uc = (r-Rc)*k2
        Vc = -np.ones(Y.shape)
    

    DXc = Uc*np.cos(Tc)+Rc*Vc*-np.sin(Tc)
    DYc = Uc*np.sin(Tc)+Rc*Vc*np.cos(Tc)
    Nc = np.sqrt(DXc** 2 + DYc** 2)
    Uc = np.nan_to_num(DXc / Nc)
    Vc = np.nan_to_num(DYc/ Nc)
    
    return X, Y, Uc+Uo, Vc+Vo

def draw_mixed(xlim, ylim, y_init, integration_t, k1, k2, po, pc, r_obstacle, 
               rotation, r):
    
    # Vector field plot
    X, Y, U, V = vector_field_mixed(xlim, ylim, k1, r_obstacle, 
                                    k2, rotation, r, po, pc)

    # Integration time
    t0 = 0.0
    tEnd = integration_t   
    t = np.linspace(t0, tEnd, round(integration_t *10))
    
    # Solution curves
    y = odeint(vf_mixed, y_init, t, args=(k1, r_obstacle, k2, rotation, r, po, pc))
    
    return y, X, Y, U, V    

def vf_obstacle(x, t, k, r_obstacle, po):
    
    dx = np.zeros(2)
    norm = np.sqrt((x[0]-po[0])**2+(x[1]-po[1])**2)
    if norm < r_obstacle : 
        dx[0] = np.sign(x[0]-po[0])*0.5*k*((1/norm)-(1/r_obstacle))**2
        dx[1] = np.sign(x[1]-po[1])*0.5*k*((1/norm)-(1/r_obstacle))**2
        
    else:
        dx[0] = 0
        dx[1] = 0
    
    return dx

def vector_field_obstacle(xlim, ylim, k, po, r_obstacle):
    
    # Vector field position
    grid_x = round((xlim[1]-xlim[0])*2.5)
    grid_y = round((ylim[1]-ylim[0])*2.5)
    X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], grid_x), np.linspace(ylim[0], ylim[1], grid_y))
    
    U=np.zeros(X.shape)
    V=np.zeros(Y.shape)
    
    # Vector field orientation
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            norm = np.sqrt((X[i,j]-po[0])**2+(Y[i,j]-po[1])**2)
            if norm < r_obstacle : 
                U[i,j] = np.sign(X[i,j]-po[0])*0.5*k*((1/norm)-(1/r_obstacle))**2
                V[i,j] = np.sign(Y[i,j]-po[1])*0.5*k*((1/norm)-(1/r_obstacle))**2
            
            else:
                U[i,j] = 0
                V[i,j] = 0
                
    # Normalize arrows
    N = np.sqrt(U** 2 + V ** 2)
    U = U / N
    V = V / N
    
    return X, Y, U, V

def draw_obstacle(xlim, ylim, y_init, integration_t, k, po, r_obstacle):
    # Vector field plot
    X, Y, U, V = vector_field_obstacle(xlim, ylim, k, po, r_obstacle)

    # Integration time
    t0 = 0.0
    tEnd = integration_t   
    t = np.linspace(t0, tEnd, round(integration_t *10))
    
    # Solution curves
    y = odeint(vf_obstacle, y_init, t, args=(k, r_obstacle, po))

    
    return y, X, Y, U, V

def vf_circle(x, t, k, rotation, r):
    # polar coordinates, x[0] = r, x[1] = theta
    dx = np.zeros(2)
    
    if rotation:
        dx[0] = (r-x[0])*k
        dx[1] = 1
    else:
        dx[0] = (r-x[0])*k
        dx[1] = -1
    
    return dx

def vector_field_circle(xlim, ylim, k, rotation, r, p1, p2):
    
    # Vector field position
    grid_x = round((xlim[1]-xlim[0])*2.5)
    grid_y = round((ylim[1]-ylim[0])*2.5)
    X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], grid_x), np.linspace(ylim[0], ylim[1], grid_y))
    
    R = np.sqrt((X-p1)**2+(Y-p2)**2)
    T = np.nan_to_num(np.arctan2(Y-p2,X-p1))
    
    # Vector field orientation
    if rotation:
        U = (r-R)*k
        V = 1
    else:
        U = (r-R)*k
        V = -1
           
    DX = U*np.cos(T)+R*V*-np.sin(T)
    DY = U*np.sin(T)+R*V*np.cos(T)
    N = np.sqrt(DX ** 2 + DY ** 2)
    U = DX / N
    V = DY / N
    
    return X, Y, U, V
    
def draw_circle(xlim, ylim, y_init, integration_t, k3, rotation, r, p1, p2):
    
    # Vector field plot
    X, Y, U, V = vector_field_circle(xlim, ylim, k3, rotation, r, p1, p2)

    # Integration time
    t0 = 0.0
    tEnd = integration_t   
    t = np.linspace(t0, tEnd, round(integration_t *10))
    
    # Solution curves
    y_initial = (np.sqrt((y_init[0]-p1)**2+(y_init[1]-p2)**2), np.arctan2(y_init[1]-p2,y_init[0]-p1))
    y = odeint(vf_circle, y_initial, t, args=(k3, rotation, r))
    y1=p1+y[:,0]*np.cos(y[:,1])
    y2=p2+y[:,0]*np.sin(y[:,1])
    y=np.stack((y1,y2),axis=1)
    
    return y, X, Y, U, V
          
def vf_line(x, t, k1, k2, theta, p1, p2):
    dx = np.zeros(2)
    if theta == 0:  
        dx[0] = 1
        dx[1] = -x[1]*k2
        
    elif theta == np.pi:
        dx[0] = -1
        dx[1] = -x[1]*k2
        
    elif theta == np.pi/2:
        dx[0] = (-x[0]+p2)*k2
        dx[1] = 1
        
    elif theta == -np.pi/2:
        dx[0] = (-x[0]+p2)*k2
        dx[1] = -1
        
    elif (0<theta<np.pi/2) or (0>theta>-np.pi/2): 
        c = np.tan(theta)+1e-10
        b = p2-c*p1
        dx[0] = (-x[0]+(x[1]-b)/c)*k2+1
        dx[1] = (-x[1]+c*x[0]+b)*k2+c
    
    elif (np.pi/2<theta<np.pi) or (-np.pi/2>theta>-np.pi):
        c = np.tan(theta)+1e-10
        b = p2-c*p1
        dx[0] = (-x[0]+(x[1]-b)/c)*k2-1
        dx[1] = (-x[1]+c*x[0]+b)*k2-c
       
    return dx
    
def vector_field_line(xlim, ylim, k1, k2, theta, p1, p2):
    
    c = np.tan(theta) # angle to line slope
    
    # Vector field position
    grid_x = round((xlim[1]-xlim[0])*2.5)
    grid_y = round((ylim[1]-ylim[0])*2.5)
    X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], grid_x), np.linspace(ylim[0], ylim[1], grid_y))
    
    # Vector field orientation
    if theta == 0:
        U = 1
        V = -Y
        
    elif theta == np.pi: 
        U = -1
        V = -Y
        
    elif theta == np.pi/2:
        U = -X+p2
        V = 1
        
    elif theta == -np.pi/2: 
        U = -X+p2
        V = -1
        
    elif (0<theta<np.pi/2) or (0>theta>-np.pi/2): 
        c = np.tan(theta)+1e-10 # angle to line slope
        b = p2-c*p1
        
        U = (-X+(Y-b)/c)*k2+1
        V = (-Y+c*X+b)*k2+c
    
    elif (np.pi/2<theta<np.pi) or (-np.pi/2>theta>-np.pi):
        c = np.tan(theta)+1e-10 # angle to line slope
        b = p2-c*p1
        
        U = (-X+(Y-b)/c)*k2-1
        V = (-Y+c*X+b)*k2-c
        
    # Normalize arrows
    N = np.sqrt(U ** 2 + V ** 2)
    U = (U / N)
    V = (V / N)
    
    return X, Y, U, V   


def draw_line(xlim, ylim, yinit, integration_t, k1, k2, theta, p1, p2):
    
    # Integration time
    t0 = 0.0
    tEnd = integration_t   
    t = np.linspace(t0, tEnd, round(integration_t *10))
    
    # Vector field plot
    X, Y, U, V = vector_field_line(xlim, ylim, k1, k2, theta, p1, p2)

    y_initial = (yinit[0], yinit[1])
    y = odeint(vf_line, y_initial, t, args=(k1, k2, theta, p1, p2))   
    
    return y, X, Y, U, V
     
def orientation_to_reach(x, y, MIX):
    # # code here the way of moving, could be a combination of differential 
    # # equation or only a simple one as limit cycle or line following
    
    ### LINE #####
    k1 = 2
    k2 = 0.7
    theta = np.pi/4
    p1 = 0
    p2 = 0
    pos = [x,y]
  
    
    dx1, dx2 = vf_line(pos, 0, k1, k2, theta, p1, p2)
    
    N = np.sqrt(dx1 ** 2 + dx1 ** 2)
    dx1 = (dx1 / N)
    dx2 = (dx2 / N)
    
    psi = np.arctan2(dx2,dx1)
    
    
    # #### OBSTACLE ####
    # if MIX:
    #     K1 = 0.01 # Trajectory aggressiveness
    #     r_obstacle = 1.5 # Obstacle radius
    #     po = [3.8,0] # Obstacle's center (x,y) coordinates
    #     pos = [x,y]
        
    #     dxo = vf_obstacle(pos, 0, K1, r_obstacle, po)
        
    #     # Normalize arrows
    #     No = np.sqrt(dxo[0]** 2 + dxo[1]** 2)
    #     Uo = np.nan_to_num(dxo[0]/ No)
    #     Vo = np.nan_to_num(dxo[1] / No)
    # else:
    #     Uo = 0
    #     Vo = 0
    
    # #### CIRCLE ####
    # K2 = 5 # Trajectory aggressiveness
    # rotation = 0 # Set to 0 for CW, 1 for ACW
    # r = 4 # radius of circle
    # pc = [0,0] # Circle's center (x,y) coordinates
    
    # # cartesian position to polar position
    # R = np.sqrt((x-pc[0])**2+(y-pc[1])**2)
    # T = np.nan_to_num(np.arctan2(y-pc[1],x-pc[0]))
    # pos=(R,T)
    
    # # polar derivatives in function of position
    # dx1, dx2 = vf_circle(pos, 0, K2, rotation, r)
      
    # # polar derivatives to cartesian derivatives
    # dx = dx1*np.cos(T)+R*dx2*-np.sin(T)
    # dy = dx1*np.sin(T)+R*dx2*np.cos(T)
    
    # # Normalize arrows
    # Nc = np.sqrt(dx** 2 + dy** 2)
    # Uc = np.nan_to_num(dx / Nc)
    # Vc = np.nan_to_num(dy / Nc)
    
    # psi = np.arctan2(Vc+Vo,Uc+Uo)
    
    return psi

def theoritic_traj(x1,x2, MIX):
    # Integration time
    integration_t = 100 
    
    # vector field param
    xlim=(-10,10)
    ylim=(-10,10)

    ## LINE ##
    k1 = 2
    k2 = 0.7
    theta = np.pi/4
    p1 = 0
    p2 = 0
    yinit = [x1,x2]
    y, X, Y, U, V = draw_line(xlim, ylim, yinit, integration_t, k1, k2, theta, p1, p2)
    
    
    # ## CIRCLE ##
    # k2 = 1 # Trajectory aggressiveness
    # rotation = 0 # Set to 0 for CW, 1 for ACW
    # r = 4 # diameter of circle
    # pc = [0,0] # Circle's center (x,y) coordinates
    
    # if MIX:
    #     ## OBSTACLE ##
    #     k1 = 0.01 # Trajectory aggressiveness
    #     r_obstacle = 1.5 # Obstacle radius
    #     po = [3.8,0] # Obstacle's center (x,y) coordinates
    #     y_init = [x1,x2]
    #     y, X, Y, U, V = draw_mixed(xlim, ylim, y_init, integration_t, k1, k2, po, pc, r_obstacle, rotation, r)
    # else:
    #     y, X, Y, U, V = draw_circle(xlim, ylim, y_init, integration_t, k2, rotation, r, pc[0], pc[1])
    
    return y, X, Y, U, V
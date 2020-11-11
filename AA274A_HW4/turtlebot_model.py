import numpy as np

EPSILON_OMEGA = 1e-3

def compute_dynamics(xvec, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute g, Gx, Gu
    # HINT: To compute the new state g, you will need to integrate the dynamics of x, y, theta
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.
    
    theta_t = xvec[2] + u[1] * dt
    
    if np.abs(u[1]) < EPSILON_OMEGA:
        g = np.array([xvec[0] + u[0] * (np.cos(theta_t) + np.cos(xvec[2]))/2. * dt,
                      xvec[1] + u[0] * (np.sin(theta_t) + np.sin(xvec[2]))/2. * dt,
                      theta_t])
        
        Gx = np.array([[1, 0, -u[0] * (np.sin(theta_t) + np.sin(xvec[2]))/2. * dt],
                       [0, 1,  u[0] * (np.cos(theta_t) + np.cos(xvec[2]))/2. * dt],
                       [0, 0, 1]])
        
        Gu = np.array([[np.cos(theta_t) * dt, -u[0] * np.sin(theta_t)/2. * dt**2.],
                       [np.sin(theta_t) * dt,  u[0] * np.cos(theta_t)/2. * dt**2.],
                       [0, dt]])
        
    else:
        g = np.array([xvec[0] + (u[0]/u[1]) * (np.sin(theta_t) - np.sin(xvec[2])),
                      xvec[1] - (u[0]/u[1]) * (np.cos(theta_t) - np.cos(xvec[2])),
                      theta_t])
        
        Gx = np.array([[1, 0, (u[0]/u[1]) * (np.cos(theta_t) - np.cos(xvec[2]))],
                       [0, 1, (u[0]/u[1]) * (np.sin(theta_t) - np.sin(xvec[2]))],
                       [0, 0, 1]])
        
        Gu = np.array([[ (np.sin(theta_t) - np.sin(xvec[2]))/u[1], -(u[0]/u[1]**2.) * (np.sin(theta_t) - np.sin(xvec[2])) + (u[0]/u[1]) * np.cos(theta_t) * dt],
                       [-(np.cos(theta_t) - np.cos(xvec[2]))/u[1],  (u[0]/u[1]**2.) * (np.cos(theta_t) - np.cos(xvec[2])) + (u[0]/u[1]) * np.sin(theta_t) * dt],
                       [0, dt]])


    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx
    # HINT: Calculate the pose of the camera in the world frame (x_cam, y_cam, th_cam), a rotation matrix may be useful.
    # HINT: To compute line parameters in the camera frame h = (alpha_in_cam, r_in_cam), 
    #       draw a diagram with a line parameterized by (alpha,r) in the world frame and 
    #       a camera frame with origin at x_cam, y_cam rotated by th_cam wrt to the world frame
    # HINT: What is the projection of the camera location (x_cam, y_cam) on the line r? 
    # HINT: To find Hx, write h in terms of the pose of the base in world frame (x_base, y_base, th_base)
    
    rot_b_to_w = np.array([[np.cos(x[2]), -np.sin(x[2]), x[0]],
                           [np.sin(x[2]),  np.cos(x[2]), x[1]],
                           [0, 0, 1]])

    x_cam_b = tf_base_to_camera[0]
    y_cam_b = tf_base_to_camera[1]
    th_cam_b = tf_base_to_camera[2]
    
    cam_b = np.array([x_cam_b, y_cam_b, 1])
    cam_w = np.matmul(rot_b_to_w, cam_b)
    cam_w[2] = th_cam_b + x[2]

    alpha_cam = alpha - cam_w[2]

    alpha_l = alpha - np.arctan2(cam_w[1], cam_w[0])
    r_cam = r - np.sqrt(cam_w[0]**2 + cam_w[1]**2) * np.cos(alpha_l)
    
    h = np.array([alpha_cam, r_cam])
    
    
    p3x = (x[0] + x_cam_b * np.cos(cam_w[2]) - y_cam_b * np.sin(cam_w[2]))**2
    p4x = x[1] + y_cam_b * np.cos(cam_w[2]) + x_cam_b * np.sin(cam_w[2])
    p2x = alpha - np.arctan2(p4x, (x[0] + x_cam_b*np.cos(cam_w[2]) - y_cam_b * np.sin(cam_w[2])))
    p1x = np.sqrt(p4x**2 + p3x)
    
    Hxx = np.sin(p2x) * p1x * p4x/(((p4x**2/p3x) + 1)*p3x) - (np.cos(p2x)*(2*x[0]+2*x_cam_b*np.cos(th_cam_b+x[2]) - 2*y_cam_b*np.sin(th_cam_b+x[2])))/(2*p1x)

    p3y = x[0] + x_cam_b*np.cos(th_cam_b+x[2]) - y_cam_b*np.sin(th_cam_b+x[2])
    p2y = alpha - np.arctan2(x[1]+y_cam_b*np.cos(th_cam_b+x[2])+x_cam_b*np.sin(th_cam_b+x[2]),p3y)
    p1y = np.sqrt(p3y**2 + (x[1]+y_cam_b*np.cos(th_cam_b+x[2])+x_cam_b*np.sin(th_cam_b+x[2]))**2) 

    Hxy = - (np.cos(p2y)*(2*x[1]+2*y_cam_b*np.cos(th_cam_b+x[2]) + 2*x_cam_b*np.sin(th_cam_b+x[2]))/(2*p1y)) - (np.sin(p2y)*p1y)/(((((x[1]+y_cam_b*np.cos(th_cam_b+x[2])+x_cam_b*np.sin(th_cam_b+x[2]))**2)/(p3y**2))+1)*p3y)      
    
    p6t = x[0] + x_cam_b*np.cos(cam_w[2]) - y_cam_b*np.sin(cam_w[2])
    p5t = x[1] + y_cam_b*np.cos(cam_w[2]) + x_cam_b*np.sin(cam_w[2])
    p4t = alpha - np.arctan2(p5t, p6t)
    p3t = np.sqrt(p6t**2 + p5t**2)
    p2t = y_cam_b * np.cos(cam_w[2]) + x_cam_b * np.sin(cam_w[2])
    p1t = x_cam_b * np.cos(cam_w[2]) - y_cam_b * np.sin(cam_w[2])
    
    Hxth = (np.cos(p4t)*(2*p2t*p6t-2*p1t*p5t)/(2*p3t)) - ((np.sin(p4t)*((p1t/p6t) + (p2t*p5t)/(p6t**2)))*p3t)/(((p5t**2)/(p6t**2))+1)
    
    Hx = np.array([[0, 0, -1], [Hxx, Hxy, Hxth]])    
    
    ########## Code ends here ##########

    if not compute_jacobian:
        return h

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h

import numpy as np 
import range_libc
from rclpy.node import Node 
import rclpy 
from tf_transformations import euler_from_quaternion
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float64
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, MapMetaData, OccupancyGrid
from nav_msgs.srv import GetMap
# Downsampling methods 
from .lidar_downsampling.box_downsampling import BoxDownsampling
from .lidar_downsampling.uniform_downsampling import UniformDownsampling

'''
These flags indicate several variants of the sensor model. Only one of them is used at a time.
'''
VAR_NO_EVAL_SENSOR_MODEL = 0
VAR_CALC_RANGE_MANY_EVAL_SENSOR = 1
VAR_REPEAT_ANGLES_EVAL_SENSOR = 2
VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT = 3
VAR_RADIAL_CDDT_OPTIMIZATIONS = 4


class ParticleFilter(Node):
    def __init__(self):
        super().__init__(
            node_name='particle_filter', 
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True
        )
                
        # ROS parameters
        self.update_rate = self.get_parameter('update_rate').get_parameter_value().integer_value
        '''Particle filter uprate rate in Hz'''
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        '''Subscribed odometry topic name'''
        self.steering_angle_topic = self.get_parameter('steering_angle_topic').get_parameter_value().double_value
        '''Subscribed steering angle topic name'''
        self.scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        '''Subscribed laserscan topic name'''
        self.pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        '''Published pose topic name'''
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        '''Base frame name'''
        self.publish_tf = self.get_parameter('publish_tf').get_parameter_value().bool_value
        '''Publish transformation between map and base_link'''
        
        
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        '''Number of particles for the particle filter'''
        self.lidar_ranges: np.ndarray = None
        '''NDarray with range measurements'''
        self.lidar_range_min: float = self.get_parameter('lidar_range_min').get_parameter_value().double_value
        '''Minimum scan range to be considered. Use -1.0 to use the laser’s reported minimum range'''
        self.lidar_range_max: float = self.get_parameter('lidar_range_max').get_parameter_value().double_value        
        '''Maximum scan range to be considered. Use -1.0 to use the laser’s reported maximum range'''
        self.lidar_range_max_px: int = None
        '''Maximum scan range in pixels'''  
        
        self.rangelib_variant = self.get_parameter('rangelib_variant').get_parameter_value().integer_value
        '''Rangelib variant'''
        self.z_hit: float = self.get_parameter('z_hit').get_parameter_value().double_value
        '''TODO'''      
        self.z_short: float = self.get_parameter('z_short').get_parameter_value().double_value
        '''TODO'''
        self.z_max: float = self.get_parameter('z_max').get_parameter_value().double_value
        '''TODO'''
        self.z_rand: float = self.get_parameter('z_rand').get_parameter_value().double_value
        '''TODO'''
        self.sigma_hit: float = self.get_parameter('simga_hit').get_parameter_value().double_value 
        '''TODO'''
        self.lamda_short: float = self.get_parameter('lambda_short').get_parameter_value().double_value
        '''TODO'''      
        self.sensor_model_lut: np.ndarray = None
        '''TODO: precomputed LUT for beam-based sensor model'''
        
        self.use_init_pose = self.get_parameter('use_initial_pose').get_parameter_value().bool_value
        '''Use initial pose for localization'''
        self.init_pose_x = self.get_parameter('initial_pose_x').get_parameter_value().double_value
        self.init_pose_y = self.get_parameter('initial_pose_y').get_parameter_value().double_value
        self.init_pose_a = self.get_parameter('initial_pose_a').get_parameter_value().double_value
        '''Initial pose (x,y,a)'''
        self.init_cov_xx = self.get_parameter('initial_cov_xx').get_parameter_value().double_value
        self.init_cov_yy = self.get_parameter('initial_cov_yy').get_parameter_value().double_value
        self.init_cov_aa = self.get_parameter('initial_cov_aa').get_parameter_value().double_value
        '''Initial covariance (xx,yy,aa)'''
        
        
        self.raycasting_id = self.get_parameter('raycasting_method').get_parameter_value().string_value
        '''Identifier to select the raycasting method'''
        self.raycasting_method = None
        '''Raycasting implementation'''
        self.theta_discretization = self.get_parameter('theta_discretization').get_parameter_value().integer_value
        '''Number of discrete bins for angular values for the (P)CDDT and LUT methods'''
        # TODO: ascontiguousarray()
        
        
        self.motion_model = self.get_parameter('motion_mo   del').get_parameter_value().string_value
        '''Motion model'''
        

        self.lidar_initialized: bool = False 
        '''Boolean flag set when laser range measurement arrived'''
        self.downsampling_initialized: bool = False
        '''Boolean flag set when lidar downsampling initialized'''
        self.odom_initialized: bool = False
        '''Boolean flag set when odometry measurement arrived'''
        self.map_initialized: bool = False
        '''Boolean flag set when occupany grid map arrived'''


        self.map_info: MapMetaData = None
        '''Map meta info'''
        self.omap: np.ndarray = None
        '''NDarray with occupancy grid map'''
        self.permissible_region: np.ndarray = None
        '''NDarray with idx of all free cells in occupancy grid map'''
        
        
        self.downsampling_method = None
        '''Lidar downsampling implementation'''
        self.downsampling_id = self.get_parameter('downsampling_method').get_parameter_value().string_value
        '''Identifier to select the downsampling method'''
        self.box_aspect_ratio = self.get_parameter('box.box_aspect_ratio').get_parameter_value().double_value
        '''Aspect ratio for box downsampling'''
        self.num_boxed_beams = self.get_parameter('box.num_beams').get_parameter_value().integer_value
        '''Number of beams to be used in box downsampling'''
        self.num_uniform_beams = self.get_parameter('uni.num_beams').get_parameter_value().integer_value
        '''Number of beams to be used in uniform downsampling'''
        
 
        # Subscriptions
        init_pose_qos_ = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,         
            depth=1,                                    
            reliability=QoSReliabilityPolicy.RELIABLE,  
            durability=QoSDurabilityPolicy.VOLATILE     
        )
        self.init_pose_sub_ = self.create_subscription(PoseWithCovarianceStamped, '/initial_pose', self.initial_pose_cb, init_pose_qos_)
        '''Subsciption to /intial_pose to reset the pose in Rviz'''
        
        sensor_qos_ = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2
        )
        self.odom_sub_ = self.create_subscription(Odometry, self.odom_topic, self.odom_cb, sensor_qos_)
        self.steering_anlge_sub_ = self.create_subscription(Float64, self.steering_angle_topic, self.steering_angle_cb, sensor_qos_)
        self.scan_sub_ = self.create_subscription(LaserScan, self.scan_topic, self.lidar_cb, sensor_qos_)    
        '''Subsciptions to sensoring data'''
               
        # Loop function
        self.timer = self.create_timer(1/self.update_rate, self.loop)
        '''Timer to perform the localization update at specific rate'''
                     
    # DONE  
    def get_map(self):
        # Use GetMap service from nav_msgs to fetch the occupany grid map
        map_client = self.create_client(GetMap, '/map_server/map')
        while not map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Get map service not available, waiting...')
        request = GetMap.Request()
        future = map_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        map_msg: OccupancyGrid = future.result().map
        
        # Extract occupany grid and map meta info
        pyomap = range_libc.PyOMap(map_msg)
        self.omap = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))
        self.map_info: MapMetaData = map_msg.info
        self.lidar_range_max_px = int(self.lidar_range_max/self.map_info.resolution)
    
        # Initialize range method with range_libc libary
        if self.raycasting_id == "BL":
            self.raycasting_method = range_libc.PyBresenhamsLine(pyomap, self.lidar_range_max_px)
        elif self.raycasting_id == "RM":
            self.raycasting_method = range_libc.PyRayMarching(pyomap, self.lidar_range_max_px)
        elif self.raycasting_id == "RMGPU":
            self.raycasting_method = range_libc.PyRayMarchingGPU(pyomap, self.lidar_range_max_px)
        elif self.raycasting_id == "GLT":
            self.raycasting_method = range_libc.PyGiantLUTCast(pyomap, self.lidar_range_max_px, self.theta_discretization)
        elif self.raycasting_id == "CDDT":
            self.raycasting_method = range_libc.PyCDDTCast(pyomap, self.lidar_range_max_px, self.theta_discretization)
        elif self.raycasting_id == "PCDDT":
            self.raycasting_method = range_libc.PyCDDTCast(pyomap, self.lidar_range_max_px, self.theta_discretization)
            self.raycasting_method.prune()
        
        # Permissible regions
        self.permissible_region = np.zeros_like(self.omap, dtype=bool)
        self.permissible_region[self.omap == 0] = 1 # marked with 1
        
        self.map_initialized = True

    # DONE
    def precompute_sensor_model(self):
        # TODO: attach derivation in pdf (see iPad)
        # TODO: link to beam-based model in Thrun et al. 2006 Probabilistic Robotics 
        
        # Extract parameters for beam-based sensor model
        z_hit = self.z_hit
        z_short = self.z_short
        z_max = self.z_max
        z_rand = self.z_rand
        sigma_hit = self.sigma_hit/self.map_info.resolution
        lambda_short = self.lamda_short/self.map_info.resolution
        
        # Create lookup-table
        lut_width = int(self.lidar_range_max_px)+1
        self.sensor_model_lut = np.zeros(shape=(lut_width, lut_width))
        
        # vectors for measured range (zm) and expected range (ze)
        zm=np.arange(0, lut_width, 1)
        ze=np.arange(0, lut_width, 1)

        # Create matrix with possible range differences (zm-ze)
        ZM = zm.reshape(-1,+1)*np.ones(lut_width).reshape(+1,-1)
        ZE = ze.reshape(+1,-1)*np.ones(lut_width).reshape(-1,+1)
        Z = ZM-ZE

        # P hit
        P_hit = (np.sqrt(2*np.pi)*sigma_hit)*np.exp(-Z**2/(2*sigma_hit**2))
        P_hit = P_hit/np.sum(P_hit, axis=0)
        
        # P short
        T_short = np.where(ZM>ZE, 0, 1)
        P_short = lambda_short*np.exp(-lambda_short*ZM)*T_short
        P_short = P_short/np.sum(P_short, axis=0)

        # P max
        P_max = np.zeros(shape=(lut_width, lut_width))
        P_max[-1,:] = 1

        # P rand 
        P_rand = np.ones(shape=(lut_width, lut_width))
        P_rand[-1,:] = 0
        P_rand = P_rand/self.lidar_range_max_px

        # Compute lookup-table for sensor model
        self.sensor_model_lut = z_hit*P_hit + z_short*P_short + z_max*P_max + z_rand*P_rand
        if self.rangelib_variant > 0:
            self.raycasting_method.set_sensor_model(self.sensor_model_lut)        
         

    def initial_pose_cb(self, msg: PoseWithCovarianceStamped):
        pass
    
    
    def odom_cb(self, msg: Odometry):
        # Extract odometry pose
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        _,_,a = euler_from_quaternion([msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z])
        self.odom_pose = np.array([x,y,a])
        
        # Initialize odometry 
        if not self.odom_initialized:
            self.odom_initialized = True
            
            
    def steering_angle_cb(self, msg: Float64):
        pass 
    
    
    def precompute_lidar_downsampling(self):
        # Boxed downsampling
        if self.downsampling_id == 'BOX':
            downsampler = BoxDownsampling(
                self.box_aspect_ratio,
                self.num_boxed_beams, 
                self.num_beams, 
                self.lidar_angle_min, 
                self.lidar_angle_max
            )
        # Uniform downsampling
        else:
            downsampler = UniformDownsampling(
                self.num_beams,
                self.num_uniform_beams
            )
        self.lidar_sampled_idx = downsampler.get_sampled_idx()
        self.downsampling_initialized = True
        
         
    def lidar_cb(self, msg: LaserScan):        
        if self.lidar_initialized and self.downsampling_initialized:
            self.lidar_ranges = np.array(msg.ranges)[self.lidar_sampled_idx]
        else:
            self.num_beams = len(msg.ranges)
            self.lidar_angle_min = msg.angle_min
            self.lidar_angle_max = msg.angle_max
            if self.lidar_range_max == -1:
                self.lidar_range_max = msg.range_max
            if self.lidar_range_min == -1: 
                self.lidar_range_min = msg.range_min
            self.lidar_initialized = True 
            

    def loop(self):
        # wait until lidar is initialized
        if not self.lidar_initialized:
            return 
        
        # do lidar downsampling init 
        if not self.downsampling_initialized: 
            self.precompute_lidar_downsampling()
                    
        # map init and precompute sensor model
        if not self.map_initialized:
            self.get_map()
            self.precompute_sensor_model()

        # main loop
        pass
        
def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
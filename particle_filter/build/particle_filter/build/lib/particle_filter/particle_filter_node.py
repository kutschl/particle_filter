import numpy as np 
import range_libc
from rclpy.node import Node 
import rclpy 
from tf_transformations import euler_from_quaternion
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, MapMetaData, OccupancyGrid
from nav_msgs.srv import GetMap
# Downsampling methods 
from .lidar_downsampling.box_downsampling import BoxDownsampling
from .lidar_downsampling.uniform_downsampling import UniformDownsampling



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
        self.scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        '''Subscribed laserscan topic name'''
        self.pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        '''Published pose topic name'''
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        '''Base frame name'''
        
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        '''Number of particles for the particle filter'''
        self.publish_tf = self.get_parameter('publish_tf').get_parameter_value().bool_value
        '''Publish transformation between map and base_link'''
        
        self.downsampling_method = self.get_parameter('downsampling_method').get_parameter_value().string_value
        '''Downsampling method to be used: "BOX" or "UNI"'''
        self.box_aspect_ratio = self.get_parameter('boxed.box_aspect_ratio').get_parameter_value().double_value
        '''Aspect ratio for box downsampling'''
        self.num_boxed_beams = self.get_parameter('boxed.num_beams').get_parameter_value().integer_value
        '''Number of beams to be used in box downsampling'''
        self.num_uniform_beams = self.get_parameter('uniform.num_beams').get_parameter_value().integer_value
        '''Number of beams to be used in uniform downsampling'''
        

        
        self.use_init_pose = self.get_parameter('use_initial_pose').get_parameter_value().bool_value
        '''Use initial pose for localization'''
        self.init_pose_x = self.get_parameter('initial_pose.position.x').get_parameter_value().double_value
        self.init_pose_y = self.get_parameter('initial_pose.position.y').get_parameter_value().double_value
        self.init_pose_a = self.get_parameter('initial_pose.orientation.a').get_parameter_value().double_value
        '''Initial pose (x,y,a)'''
        self.init_cov_xx = self.get_parameter('initial_pose.covariance.xx').get_parameter_value().double_value
        self.init_cov_yy = self.get_parameter('initial_pose.covariance.yy').get_parameter_value().double_value
        self.init_cov_aa = self.get_parameter('initial_pose.covariance.aa').get_parameter_value().double_value
        '''Diagonal elements of initial covariance matrix (xx,yy,aa)'''
        
        self.raycasting_method = self.get_parameter('raycasting_method').get_parameter_value().string_value
        '''Raycasting method'''
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
        
        
        self.downsampling = None
        '''Lidar downsampling instance'''
        self.lidar_ranges: np.ndarray = None
        '''NDarray with range measurements'''
        self.lidar_range_min: float = self.get_parameter('lidar_range_min').get_parameter_value().double_value
        '''Minimum scan range to be considered. Use -1.0 to use the laser’s reported minimum range'''
        self.lidar_range_max: float = self.get_parameter('lidar_range_max').get_parameter_value().double_value        
        '''Maximum scan range to be considered. Use -1.0 to use the laser’s reported maximum range'''
        self.lidar_range_max_px: int = None
        '''Maximum scan range in pixels'''  
              
        # Subscriptions
        init_pose_qos_ = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,         
            depth=1,                                    
            reliability=QoSReliabilityPolicy.RELIABLE,  
            durability=QoSDurabilityPolicy.VOLATILE     
        )
        self.init_pose_sub_ = self.create_subscription(PoseWithCovarianceStamped, '/initial_pose', self.initial_pose_cb, init_pose_qos_)
        
        odom_sub_qos_ = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2
        )
        self.odom_sub_ = self.create_subscription(Odometry, self.odom_topic, self.odom_cb, odom_sub_qos_)
        
        scan_sub_qos_ = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2 
        )
        self.scan_sub_ = self.create_subscription(LaserScan, self.scan_topic, self.lidar_cb, scan_sub_qos_)    
               
        # Loop function
        self.timer = self.create_timer(1/self.update_rate, self.loop)
                     
                       
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
        if self.raycasting_method == "BL":
            self.range_method = range_libc.PyBresenhamsLine(pyomap, self.lidar_range_max_px)
        elif self.raycasting_method == "RM":
            self.range_method = range_libc.PyRayMarching(pyomap, self.lidar_range_max_px)
        elif self.raycasting_method == "RMGPU":
            self.range_method = range_libc.PyRayMarchingGPU(pyomap, self.lidar_range_max_px)
        elif self.raycasting_method == "GLT":
            self.range_method = range_libc.PyGiantLUTCast(pyomap, self.lidar_range_max_px, self.THETA_DISCRETIZATION)
        # TODO: PCDDT, CDDT
        
        # Permissible regions
        self.permissible_region = np.zeros_like(self.omap, dtype=bool)
        self.permissible_region[self.omap == 0] = 1
        
        self.map_initialized = True

    
    
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
    
    
    def precompute_lidar_downsampling(self):
        # Boxed downsampling
        if self.downsampling_method == 'BOX':
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
            self.lidar_ranges = self.downsampling(np.array(msg.ranges))
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
                    
        # map init 
        if not self.map_initialized:
            self.get_map()
        
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
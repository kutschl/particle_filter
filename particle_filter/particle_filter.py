# TODO: check irrelevant imports 
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import IntegerRange, FloatingPointRange
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
import tf_transformations as tft
import numpy as np
import range_libc
from threading import Lock
from .utils import utils as Utils

# Message types
from geometry_msgs.msg import PoseStamped, PoseArray, PointStamped, \
    PoseWithCovarianceStamped, TransformStamped, Quaternion
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, MapMetaData, OccupancyGrid
from nav_msgs.srv import GetMap

# Define Datatypes
from enum import Enum
from typing import Optional

# TODO: remove debug stuff
# Debug
from time import time
from collections import deque
import cProfile
import pstats

# TODO: # Dynamic Reconfigure
# from dynamic_reconfigure.msg import Config


class RangeLibVariant(Enum):
    '''
    These flags indicate several variants of the sensor model. Only one of them is used at a time.
    '''
    VAR_NO_EVAL_SENSOR_MODEL = 0
    VAR_CALC_RANGE_MANY_EVAL_SENSOR = 1
    VAR_REPEAT_ANGLES_EVAL_SENSOR = 2
    VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT = 3
    VAR_RADIAL_CDDT_OPTIMIZATIONS = 4
    

class ParticleFilter(Node):
    '''
    Particle Filter Two: Electric Boogaloo.

    A refactor of the MIT Racecar Particle Filter code with augmentations from other places,
    notably the TUM paper "ROS-based localization of a race vehicle at high-speed using LIDAR".
    '''
    
    def __init__(self):
        super().__init__('particle_filter')
        
        self.node_initialized = False
        
        # Parameter 
        self.declare_parameter(
            "max_particles",
            3000,
            ParameterDescriptor(
                description="Maximum number of particles in the particle filter. Suitable values: <=3000",
                integer_range=[IntegerRange(from_value=1, to_value=5000)]
            )
        )
        self.declare_parameter(
            "max_viz_particles",
            50,
            ParameterDescriptor(
                description="Number of particles to visualize if visualization is enabled.",
                integer_range=[IntegerRange(from_value=1, to_value=5000)]
            )
        )
        self.declare_parameter(
            "squash_factor",
            2.2,
            ParameterDescriptor(
                description="Probability adjustment factor for the sensor model.",
            )
        )
        self.declare_parameter(
            "max_range",
            20.0,
            ParameterDescriptor(
                description="Maximum range of the rangefinder sensor.",
            )
        )
        self.declare_parameter(
            "range_method",
            "glt",
            ParameterDescriptor(
                description="Method for range calculation (e.g., 'glt', 'rmgpu', 'pcddt')."
            )
        )
        self.declare_parameter(
            "rangelib_variant",
            3,
            ParameterDescriptor(
                description=(
                    "RangeLib variant to use for range calculations. Allowed values:\n"
                    "0: VAR_NO_EVAL_SENSOR_MODEL - No sensor model evaluation\n"
                    "1: VAR_CALC_RANGE_MANY_EVAL_SENSOR - Calculate range with multiple evaluations\n"
                    "2: VAR_REPEAT_ANGLES_EVAL_SENSOR - Repeat angles with sensor model evaluation\n"
                    "3: VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT - One-shot evaluation with repeated angles\n"
                    "4: VAR_RADIAL_CDDT_OPTIMIZATIONS - Radial optimizations for CDDT"
                ),
                integer_range=[IntegerRange(from_value=0, to_value=4)]
            )
        )
        self.declare_parameter(
            "pose_pub_topic",
            "/pf/pose",
            ParameterDescriptor(
                description="Topic for publishing the tracked pose."
            )
        )
        self.declare_parameter(
            "use_viz",
            False,
            ParameterDescriptor(
                description="Enable or disable visualization."
            )
        )
        self.declare_parameter(
            "publish_tf",
            True,
            ParameterDescriptor(
                description="Enable publishing of TF transforms."
            )
        )
        self.declare_parameter(
            "odom_topic",
            "/odom",
            ParameterDescriptor(
                description="Topic name for odometry data."
            )
        )
        self.declare_parameter(
            "scan_topic",
            "/scan",
            ParameterDescriptor(
                description="Topic name for laserscan data."
            )
        )
        self.declare_parameter(
            "map_topic",
            "/map",
            ParameterDescriptor(
                description="Topic name for occupancy grid map."
            )
        )
        self.declare_parameter(
            "theta_discretization",
            150,
            ParameterDescriptor(
                description="Number of discrete angular values for the lookup table (LUT)."
            )
        )
        self.declare_parameter(
            "publish_covariance",
            False,
            ParameterDescriptor(
                description="Publish empirically calculated covariance."
            )
        )
        self.declare_parameter(
            "use_initial_pose",
            True,
            ParameterDescriptor(description="Whether to use the specified initial pose for the particle filter. true: initial pose used, false: global localization.")
        )
        self.declare_parameter(
            "initial_pose_x",
            0.0,
            ParameterDescriptor(description="Initial x position of the particle filter.")
        )
        self.declare_parameter(
            "initial_pose_y",
            0.0,
            ParameterDescriptor(description="Initial y position of the particle filter.")
        )
        self.declare_parameter(
            "initial_pose_theta",
            0.0,
            ParameterDescriptor(description="Initial orientation (theta) of the particle filter.")
        )
        self.declare_parameter(
            "initial_var_x",
            0.5,
            ParameterDescriptor(description="Initial variance in the x direction.")
        )
        self.declare_parameter(
            "initial_var_y",
            0.5,
            ParameterDescriptor(description="Initial variance in the y direction.")
        )
        self.declare_parameter(
            "initial_var_theta",
            0.4,
            ParameterDescriptor(description="Initial variance in the theta (orientation) direction.")
        )
        self.declare_parameter(
            "z_hit",
            0.75,
            ParameterDescriptor(description="Probability of hitting the intended target.")
        )
        self.declare_parameter(
            "z_short",
            0.01,
            ParameterDescriptor(description="Probability of an unexpected short reading.")
        )
        self.declare_parameter(
            "z_max",
            0.07,
            ParameterDescriptor(description="Probability of an out-of-range reading beyond max_range.")
        )
        self.declare_parameter(
            "z_rand",
            0.12,
            ParameterDescriptor(description="Probability of a random reading anywhere in the valid range.")
        )
        self.declare_parameter(
            "sigma_hit",
            0.4,
            ParameterDescriptor(description="Standard deviation of the hit probability distribution.")
        )
        self.declare_parameter(
            "lambda_short",
            0.1,
            ParameterDescriptor(description="Parameter for the short reading exponential distribution.")
        )
        self.declare_parameter(
            "motion_model",
            "tum",
            ParameterDescriptor(description="Motion model to use. Allowed values: 'tum', 'amcl', 'arc'.")
        )
        self.declare_parameter(
            "alpha_1_tum",
            0.5,
            ParameterDescriptor(description="Effect of rotation on rotation variance in TUM motion model.")
        )
        self.declare_parameter(
            "alpha_2_tum",
            0.015,
            ParameterDescriptor(description="Effect of translation on rotation variance in TUM motion model.")
        )
        self.declare_parameter(
            "alpha_3_tum",
            1.0,
            ParameterDescriptor(description="Effect of translation on translation variance in TUM motion model.")
        )
        self.declare_parameter(
            "alpha_4_tum",
            0.1,
            ParameterDescriptor(description="Effect of rotation on translation variance in TUM motion model.")
        )
        self.declare_parameter(
            "lambda_thresh",
            0.1,
            ParameterDescriptor(description="Minimum translation between frames for the TUM model to become effective.")
        )
        self.declare_parameter(
            "alpha_1_amcl",
            0.5,
            ParameterDescriptor(description="Effect of rotation on rotation variance in AMCL motion model.")
        )
        self.declare_parameter(
            "alpha_2_amcl",
            0.5,
            ParameterDescriptor(description="Effect of translation on rotation variance in AMCL motion model.")
        )
        self.declare_parameter(
            "alpha_3_amcl",
            1.0,
            ParameterDescriptor(description="Effect of translation on translation variance in AMCL motion model.")
        )
        self.declare_parameter(
            "alpha_4_amcl",
            0.1,
            ParameterDescriptor(description="Effect of rotation on translation variance in AMCL motion model.")
        )
        self.declare_parameter(
            "motion_dispersion_arc_x",
            0.05,
            ParameterDescriptor(description="Motion dispersion in x direction for ARC model."),
        )
        self.declare_parameter(
            "motion_dispersion_arc_y",
            0.025,
            ParameterDescriptor(description="Motion dispersion in y direction for ARC model.")
        )
        self.declare_parameter(
            "motion_dispersion_arc_theta",
            0.25,
            ParameterDescriptor(description="Motion dispersion in theta direction for ARC model.")
        )
        self.declare_parameter(
            "motion_dispersion_arc_xy",
            0.0,
            ParameterDescriptor(description="Motion dispersion between x and y for ARC model.")
        )
        self.declare_parameter(
            "motion_dispersion_arc_x_min",
            0.01,
            ParameterDescriptor(description="Minimum dispersion in x direction for ARC model.")
        )
        self.declare_parameter(
            "motion_dispersion_arc_y_min",
            0.01,
            ParameterDescriptor(description="Minimum dispersion in y direction for ARC model.")
        )
        self.declare_parameter(
            "motion_dispersion_arc_y_max",
            0.01,
            ParameterDescriptor(description="Maximum dispersion in y direction for ARC model.")
        )
        self.declare_parameter(
            "motion_dispersion_arc_theta_min",
            0.01,
            ParameterDescriptor(description="Minimum dispersion in theta direction for ARC model.")
        )
        self.declare_parameter(
            "motion_dispersion_arc_xy_min_x",
            0.01,
            ParameterDescriptor(description="Minimum dispersion between x and y for ARC model.")
        )
        self.declare_parameter(
            "motion_dispersion_x",
            0.05,
            ParameterDescriptor(description="Motion dispersion in the x direction for default MIT model.")
        )
        self.declare_parameter(
            "motion_dispersion_y",
            0.025,
            ParameterDescriptor(description="Motion dispersion in the y direction for default MIT model.")
        )
        self.declare_parameter(
            "motion_dispersion_theta",
            0.25,
            ParameterDescriptor(description="Motion dispersion in the theta (orientation) direction for default MIT model.")
        )
        self.declare_parameter(
            "lidar_aspect_ratio",
            3.0,
            ParameterDescriptor(description="Aspect ratio of the LiDAR sensor field.")
        )
        self.declare_parameter(
            "des_lidar_beams",
            21,
            ParameterDescriptor(description="Desired number of LiDAR beams. Should be an odd number.")
        )
        self.declare_parameter(
            'tf_base_link_to_laser_x',
            0.27,
            ParameterDescriptor(description="Transformation translation x-axis between base_link and laser frame.")       
        )
        self.declare_parameter(
            'tf_base_link_to_laser_y',
            0.00,
            ParameterDescriptor(description="Transformation translation y-axis between base_link and laser frame.")       
        )
        self.declare_parameter(
            'tf_base_link_to_laser_z',
            0.11,
            ParameterDescriptor(description="Transformation translation z-axis between base_link and laser frame.")       
        )

        # General
        self.max_particles = self.get_parameter("max_particles").get_parameter_value().integer_value
        self.max_viz_particles = self.get_parameter("max_viz_particles").get_parameter_value().integer_value
        self.inv_squash_factor = 1.0 / self.get_parameter("squash_factor").get_parameter_value().double_value
        self.max_range = self.get_parameter("max_range").get_parameter_value().double_value
        self.range_method = self.get_parameter("range_method").get_parameter_value().string_value.lower()
        self.rangelib_variant = RangeLibVariant(self.get_parameter("rangelib_variant").get_parameter_value().integer_value)
        self.pose_pub_topic = self.get_parameter("pose_pub_topic").get_parameter_value().string_value
        self.use_viz = self.get_parameter("use_viz").get_parameter_value().bool_value
        self.publish_tf = self.get_parameter("publish_tf").get_parameter_value().bool_value
        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.map_topic = self.get_parameter("map_topic").get_parameter_value().string_value
        self.theta_discretization = self.get_parameter("theta_discretization").get_parameter_value().integer_value
        self.publish_covariance = self.get_parameter("publish_covariance").get_parameter_value().bool_value
        # TODO: in the future use tfs lookup for base_link to laser tf
        self.tf_base_link_to_laser_x = self.get_parameter("tf_base_link_to_laser_x").get_parameter_value().double_value
        self.tf_base_link_to_laser_y = self.get_parameter("tf_base_link_to_laser_y").get_parameter_value().double_value
        self.tf_base_link_to_laser_z = self.get_parameter("tf_base_link_to_laser_z").get_parameter_value().double_value
        
        # (Re) initialization constants
        self.use_init_pose = self.get_parameter("use_initial_pose").get_parameter_value().bool_value
        self.init_pose_x = self.get_parameter("initial_pose_x").get_parameter_value().double_value
        self.init_pose_y = self.get_parameter("initial_pose_y").get_parameter_value().double_value
        self.init_pose_theta = self.get_parameter("initial_pose_theta").get_parameter_value().double_value
        self.init_var_x = self.get_parameter("initial_var_x").get_parameter_value().double_value
        self.init_var_y = self.get_parameter("initial_var_y").get_parameter_value().double_value
        self.init_var_theta = self.get_parameter("initial_var_theta").get_parameter_value().double_value
        
        # Sensor model constants
        self.z_hit = self.get_parameter("z_hit").get_parameter_value().double_value
        self.z_short = self.get_parameter("z_short").get_parameter_value().double_value
        self.z_max = self.get_parameter("z_max").get_parameter_value().double_value
        self.z_rand = self.get_parameter("z_rand").get_parameter_value().double_value
        self.sigma_hit = self.get_parameter("sigma_hit").get_parameter_value().double_value
        self.lambda_short = self.get_parameter("lambda_short").get_parameter_value().double_value
        
        # Motion model constants
        self.motion_model_param = self.get_parameter("motion_model").get_parameter_value().string_value.lower()
        if self.motion_model_param == 'tum':
            self.alpha_1 = self.get_parameter("alpha_1_tum").get_parameter_value().double_value
            self.alpha_2 = self.get_parameter("alpha_2_tum").get_parameter_value().double_value
            self.alpha_3 = self.get_parameter("alpha_3_tum").get_parameter_value().double_value
            self.alpha_4 = self.get_parameter("alpha_4_tum").get_parameter_value().double_value
            self.lambda_thresh = self.get_parameter("lambda_thresh").get_parameter_value().double_value
            self.get_logger().info("Using TUM motion model")
        elif self.motion_model_param == 'amcl':
            self.alpha_1 = self.get_parameter("alpha_1_amcl").get_parameter_value().double_value
            self.alpha_2 = self.get_parameter("alpha_2_amcl").get_parameter_value().double_value
            self.alpha_3 = self.get_parameter("alpha_3_amcl").get_parameter_value().double_value
            self.alpha_4 = self.get_parameter("alpha_4_amcl").get_parameter_value().double_value
            self.get_logger().info("Using AMCL motion model")
        elif self.motion_model_param == 'arc':
            self.motion_dispersion_arc_x = self.get_parameter("motion_dispersion_arc_x").get_parameter_value().double_value
            self.motion_dispersion_arc_y = self.get_parameter("motion_dispersion_arc_y").get_parameter_value().double_value
            self.motion_dispersion_arc_theta = self.get_parameter("motion_dispersion_arc_theta").get_parameter_value().double_value
            self.motion_dispersion_arc_xy = self.get_parameter("motion_dispersion_arc_xy").get_parameter_value().double_value
            self.motion_dispersion_arc_x_min = self.get_parameter("motion_dispersion_arc_x_min").get_parameter_value().double_value
            self.motion_dispersion_arc_y_min = self.get_parameter("motion_dispersion_arc_y_min").get_parameter_value().double_value
            self.motion_dispersion_arc_y_max = self.get_parameter("motion_dispersion_arc_y_max").get_parameter_value().double_value
            self.motion_dispersion_arc_theta_min = self.get_parameter("motion_dispersion_arc_theta_min").get_parameter_value().double_value
            self.motion_dispersion_arc_xy_min_x = self.get_parameter("motion_dispersion_arc_xy_min_x").get_parameter_value().double_value
            self.get_logger().info("Using ARC motion model")
        else:
            # TODO: change params name: ..._mit (we are using as default motion model the MIT motion model)
            self.motion_dispersion_x = self.get_parameter("motion_dispersion_x").get_parameter_value().double_value
            self.motion_dispersion_y = self.get_parameter("motion_dispersion_y").get_parameter_value().double_value
            self.motion_dispersion_theta = self.get_parameter("motion_dispersion_theta").get_parameter_value().double_value
            self.get_logger().info("Using default MIT motion model")
        
        # Boxed lidar model (using Hokuyo laser scanner) with subscription for scan params
        self.lidar_aspect_ratio = self.get_parameter("lidar_aspect_ratio").get_parameter_value().double_value
        self.des_lidar_beams = self.get_parameter("des_lidar_beams").get_parameter_value().integer_value
        self.num_lidar_beams = None
        self.start_theta = None
        self.end_theta = None
        self.scan_params_initialized = False
        '''Boolean flag set when the lidar scan parameters have been updated'''
        self.scan_sub = self.create_subscription(
            LaserScan, 
            self.scan_topic, 
            self.scan_cb,
            2
        )
        
        # Map parameters
        self.map_msg: OccupancyGrid = None 
        self.map_params_initialized = False 
        '''Boolean flag set when the map parameters have been updated'''
        from rclpy.qos import QoSProfile, QoSDurabilityPolicy
        qos_profile = QoSProfile(depth=10)
        qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_cb,
            qos_profile
        )
        
        # Data members in the Particle Filter
        self.state_lock = Lock()
        '''Lock to prevent multithreading errors'''
        self.rate = self.create_rate(200)
        '''Enforces update rate. This should not be lower than the odom message rate (50)'''
        self.max_range_px: int = None
        '''Maximum lidar range in pixels'''
        self.odometry_data = np.array([0.0, 0.0, 0.0])
        '''NDArray Buffer for odometry data (x, y, theta) representing the change in last and current odom position'''
        self.map_info: MapMetaData = None
        '''Buffer for Occupancy Grid metadata'''
        self.permissible_region: np.ndarray = None
        '''NDArray containing the OccupancyGrid. 0: not permissible, 1: permissible.'''
        self.map_initialized = False
        '''Boolean flag set when `get_omap()` is called'''
        self.lidar_initialized = False
        '''Boolean flag set when the Lidar Scan arrays have been populated'''
        self.odom_initialized = False
        '''Boolean flag set when `self.odometry_data` has been initialized'''
        self.last_pose: np.ndarray = None
        '''3-vector holding the last-known pose from odometry (x,y,theta)'''
        self.curr_pose: np.ndarray = None
        '''3-vector holding the current pose from odometry (x,y,theta)'''
        self.last_stamp: Time = Time(seconds=0.0)
        '''Timestamp of last-received odometry message'''
        self.last_pub_stamp = self.get_clock().now()
        '''Last published timestamp'''
        self.first_sensor_update = True
        '''Boolean flag for use in `sensor_model()`'''
        self.odom_msgs: deque = deque([], maxlen=5)
        '''Buffer holding the last few Odometry messages, which may be clumped for some reason.'''
        self.local_deltas = np.zeros((self.max_particles, 3))
        '''NDArray of local motion, allocated for use in motion model'''
        
        # cache these for the sensor model computation
        self.queries: np.ndarray = None
        '''NDArray of sensor queries (call to RangeLibc), init'd on the first `sensor_model()` call'''
        self.ranges: np.ndarray = None
        '''NDArray of ranges returned by RangeLibc'''
        self.tiled_angles: np.ndarray = None
        '''Used in `sensor_model()`'''
        self.sensor_model_table: np.ndarray = None
        '''NDArray containing precomputed, discretized sensor model probability init'd in `precompute_sensor_model()`'''
        self.lidar_sample_idxs: np.ndarray = None
        '''NDArray holding the evenly spaced lidar indices to sample from, calcualted in `get_boxed_indices()`'''
        self.lidar_theta_lut: np.ndarray = None
        '''Single-precision NDArray lookup table of the lidar beam angles at the indices of LIDAR_SAMPLE_IDXS'''

        # particle poses and weights
        self.inferred_pose: np.ndarray = None
        '''NDArray of the expected value of the pose given the particle distribution'''
        self.particle_indices = np.arange(self.max_particles)
        '''Numbered list of particles.'''
        self.particles = np.zeros((self.max_particles, 3))
        '''NDArray of potential particles. Each represents a hypothesis location of the base link. (MAX_PARTICLES, 3)'''
        self.weights = np.ones(self.max_particles) / float(self.max_particles)
        '''NDArray weighting the particles, initialized uniformly (MAX_PARTICLES, )'''
        if self.publish_covariance:
            self.cov = np.zeros((3, 3))
            '''NDArray representing the covariance (x,y,theta)'''
            
        # Publisher
        self.particle_pub = self.create_publisher(PoseArray, "/pf/viz/particles", 10)
        '''Publishes particle cloud onto /pf/viz/particles (randomly sampled)'''
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_pub_topic, 10)
        '''Publishes inferred pose on POSE_PUB_TOPIC (default: /pf/pose) '''
        if self.publish_covariance:
            self.pose_cov_pub = self.create_publisher(PoseWithCovarianceStamped, f'{self.pose_pub_topic}_with_cov', 10)
            '''Publishes inferred pose with Covariance. (default: `/tracked_pose/with_covariance`)'''
        
        # Subscriptions: placeholders after init_scan_sub is spinned
        self.odom_sub = None
        self.init_pose_sub = None
        self.clicked_point_sub = None
        self.map_sub = None
                
        # Transformations
        self.pub_tf = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
            
    def complete_initialization(self):
        """Completes the initialization after the scan parameters are set."""
        
        self.get_logger().info("Proceeding with the remaining initialization.")
        
        # Initialize Map and Sensor Model
        self.get_boxed_indices()
        self.get_omap()
        self.precompute_sensor_model()
        
        # Initialize localization
        if self.use_init_pose:
            self.initialize_particles_pose(self.init_pose_x, self.init_pose_y, self.init_pose_theta)
        else:
            self.initialize_global()
            
        # Offset base_link to laser
        # TODO: future work: use transformation instead of parameters 
        self.laser_base_link_offset = np.array([
            self.tf_base_link_to_laser_x,
            self.tf_base_link_to_laser_y,
            self.tf_base_link_to_laser_z
        ])
        self.particle_utils = Utils.ParticleUtils(self.laser_base_link_offset)
        
        # Initialize other subscriptions
        self.odom_sub = self.create_subscription(
            Odometry, 
            self.odom_topic, 
            self.odom_cb,
            2
        )
        self.initial_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "/initialpose",
            self.clicked_pose_cb,
            1
        )
        self.clicked_point_sub = self.create_subscription(
            PointStamped,
            "/clicked_point",
            self.clicked_pose_cb,
            1 
        )
        
        # Timer for filter update loop 
        self.localization_loop_timer = self.create_timer(
            0.5,
            self.localization_loop
        )
        
        self.get_logger().info("Initialization was successful.")
        self.node_initialized = True
                
    def scan_cb(self, msg: LaserScan):
        if not self.scan_params_initialized:
            self.num_lidar_beams = len(msg.ranges)
            self.start_theta = msg.angle_min
            self.end_theta = msg.angle_max
            self.scan_params_initialized = True
            self.get_logger().info(f"Initialized scan parameters (num_lidar_beams: {self.num_lidar_beams}, start_theta: {self.start_theta}, end_theta: {self.end_theta})")        
        elif self.scan_params_initialized and self.map_params_initialized and not self.node_initialized:
            self.complete_initialization()
        elif self.node_initialized:
            self.downsampled_ranges = np.array(msg.ranges)[self.lidar_sample_idxs]
            self.lidar_initialized = True
    
    def map_cb(self, msg: OccupancyGrid):
        if not self.map_params_initialized:
            self.map_msg = msg
            self.map_info = msg.info
            self.map_params_initialized = True
            self.get_logger().info(f"Initialized map parameters (resolution: {msg.info.resolution}") 


    def odom_cb(self, msg: Odometry):
        self.odom_msgs.append(msg)
        self.last_odom_msg = msg
        
        # TODO: code cleaning
        # rospy.loginfo(f"Odom gap: {(rospy.Time.now().to_sec()-self.last_odom)*1000}ms")
        # self.last_odom = rospy.Time.now().to_sec()

        # if self.last_odom_msg is not None:
        #     dx = self.last_odom_msg.pose.pose.position.x - msg.pose.pose.position.x
        #     dy = self.last_odom_msg.pose.pose.position.y - msg.pose.pose.position.y
        #     rospy.loginfo(f"{dx=:.6f}, {dy=:.6f}")
    
    def clicked_pose_cb(self, msg):
        '''
        Receive pose messages from RViz and initialize the particle distribution in response.
        '''
        if isinstance(msg, PointStamped):
            self.initialize_global()
        elif isinstance(msg, PoseWithCovarianceStamped):
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            o = msg.pose.pose.orientation
            self.initialize_particles_pose(x, y, poseo=o)
            
    def initialize_particles_pose(self, posex: float, posey: float,
                                  posetheta: Optional[float] = None,
                                  poseo: Optional[Quaternion] = None):
        '''
        Initialize particles in the general region of the provided pose.

        Either initialize with a yaw (theta) or Quaternion.
        '''
        assert not (posetheta is None and poseo is None)
        assert not (posetheta is not None and poseo is not None)

        init_var_x = self.init_var_x
        init_var_y = self.init_var_y
        init_var_th = self.init_var_theta
        
        while self.state_lock.locked():
            self.get_logger().info_once("PF2 Pose Initialization: Waiting for state to become unlocked")
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.1))

        
        self.state_lock.acquire()

        if poseo is not None:
            posetheta = Utils.quaternion_to_angle(poseo)

        self.get_logger().info(
            f"Setting initial pose at x:{posex:.2f}, y:{posey:.2f}, theta:{np.degrees(posetheta):.2f}deg")
        self.weights = np.ones(self.max_particles) / float(self.max_particles)
        self.particles[:, 0] = posex + \
            np.random.normal(scale=init_var_x, size=self.max_particles)
        self.particles[:, 1] = posey + \
            np.random.normal(scale=init_var_y, size=self.max_particles)
        self.particles[:, 2] = posetheta + \
            np.random.normal(scale=init_var_th, size=self.max_particles)

        self.state_lock.release()
    
    def initialize_global(self):
        '''
        Spread the particle distribution over the permissible region of the state space.

        Future Extension: Informed sampling by spreading over the race line
        '''
        while self.state_lock.locked():
            self.get_logger().info_once("PF2 Global Initialization: Waiting for state to become unlocked")
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.1))

        self.state_lock.acquire()
        self.get_logger().info("Lost Robot Initialization")

        # randomize over grid coordinate space
        permissible_x, permissible_y = np.where(self.permissible_region == 1)
        indices = np.random.randint(
            0, len(permissible_x), size=self.max_particles)

        permissible_states = np.zeros((self.max_particles, 3))
        permissible_states[:, 0] = permissible_y[indices]
        permissible_states[:, 1] = permissible_x[indices]
        permissible_states[:, 2] = np.random.random(
            self.max_particles) * np.pi * 2.0

        Utils.map_to_world(permissible_states, self.map_info)
        self.particles = permissible_states
        self.weights[:] = 1.0 / self.max_particles

        self.state_lock.release()
    
    def get_omap(self):
        '''
        Fetch the occupancy grid map from the map_server instance, and initialize the correct
        RangeLibc method. Also stores a matrix which indicates the permissible region of the map
        '''
        # rospy.wait_for_service("static_map")
        # map_msg = rospy.ServiceProxy("static_map", GetMap)().map

        # self.map_info = map_msg.info
        oMap = range_libc.PyOMap(self.map_msg)
        self.max_range_px = int(self.max_range / self.map_info.resolution)

        # initialize range method
        self.get_logger().info(f"Initializing range method: {self.range_method}")
        if self.range_method == "bl":
            self.range_method = range_libc.PyBresenhamsLine(
                oMap, self.max_range_px)
        elif "cddt" in self.range_method:
            self.range_method = range_libc.PyCDDTCast(
                oMap, self.max_range_px, self.theta_discretization)
            if self.range_method == "pcddt":
                self.get_logger().info("Pruning...")
                self.range_method.prune()
        elif self.range_method == "rm":
            self.range_method = range_libc.PyRayMarching(
                oMap, self.max_range_px)
        elif self.range_method == "rmgpu":
            self.range_method = range_libc.PyRayMarchingGPU(
                oMap, self.max_range_px)
        elif self.range_method == "glt":
            self.range_method = range_libc.PyGiantLUTCast(
                oMap, self.max_range_px, self.theta_discretization)
        self.get_logger().info("Done loading map")

        # 0: permissible, -1: unmapped, 100: blocked
        array_255 = np.array(self.map_msg.data).reshape((self.map_msg.info.height, self.map_msg.info.width))

        # 0: not permissible, 1: permissible
        self.permissible_region = np.zeros_like(array_255, dtype=bool)
        self.permissible_region[array_255 == 0] = 1

        # TODO: code cleaning // Sanity Check
        # _, axs = plt.subplots(nrows=2, ncols=1)
        # axs[0].set_title("Original data")
        # axs[0].imshow(array_255)
        # axs[1].set_title("Permissible Region")
        # im=axs[1].imshow(self.permissible_region)
        # plt.colorbar(im, orientation="horizontal")
        # plt.show()

        self.map_initialized = True
        
    def precompute_sensor_model(self):
        '''
        Generate and store a lookup table which represents the sensor model.

        For each discrete computed range value, this provides the probability of measuring that (discrete) range.

        This table is indexed by the sensor model at runtime by discretizing the measurements
        and computed ranges from RangeLibc.
        '''
        self.get_logger().info("Precomputing sensor model")
        # sensor model constants
        z_short = self.z_short
        z_max = self.z_max
        z_rand = self.z_rand
        z_hit = self.z_hit
        # normalise sigma and lambda from meters to pixel space
        # [px] = [m] / [m/px]
        sigma_hit = self.sigma_hit/self.map_info.resolution
        lam_short = self.lambda_short/self.map_info.resolution

        table_width = int(self.max_range_px) + 1
        self.sensor_model_table = np.zeros((table_width, table_width))

        # compute normalizers for the gaussian and exponential distributions
        norm_gau = np.zeros((table_width,))
        norm_exp = np.zeros((table_width,))
        for d in range(table_width):
            sum_gau = 0
            sum_exp = 0
            for r in range(table_width):
                z = float(d-r)
                sum_gau += np.exp(-(z*z)/(2.0*sigma_hit*sigma_hit)) / \
                        (sigma_hit * np.sqrt(2.0*np.pi))

                if r <= d:
                    sum_exp += ( lam_short * np.exp(-lam_short*r) )

            norm_gau[d] = 1/sum_gau
            norm_exp[d] = 1/sum_exp

        # d is the computed range from RangeLibc (predicted range)
        for d in range(table_width):
            norm = 0.0
            # r is the observed range from the lidar unit
            for r in range(table_width):
                prob = 0.0
                z = float(d-r)

                # Probability of hitting the intended object
                # P_hit -- sample from a Gaussian
                prob += z_hit * \
                    ( np.exp(-(z*z)/(2.0*sigma_hit*sigma_hit)) / \
                    (sigma_hit * np.sqrt(2.0*np.pi)) ) * norm_gau[d]

                # observed range is less than the predicted range - short reading
                # P_short -- sample from exponential distribution
                # note: z must be positive here!
                if r <= d:
                    prob += z_short * ( lam_short * np.exp(-lam_short*r) ) * norm_exp[d]

                # erroneous max range measurement
                # P_max -- uniform distribution at max range
                if r == int(self.max_range_px):
                    prob += z_max

                # random measurement
                # P_rand -- uniform distribution across entire range
                if r < self.max_range_px:
                    prob += z_rand * 1.0/self.max_range_px

                norm += prob
                self.sensor_model_table[r, d] = prob

            # normalize
            self.sensor_model_table[:, d] /= norm

        # upload the sensor model to RangeLib for acceleration
        if self.rangelib_variant.value > 0:
            self.range_method.set_sensor_model(self.sensor_model_table)

    def get_boxed_indices(self):
        '''
        Finds an evenly spaced "boxed" pattern of beams based on the TUM paper
        "ROS-based localization of a race vehicle at high-speed using LIDAR".
        '''
        beam_angles = np.linspace(
            self.start_theta, self.end_theta, self.num_lidar_beams)

        MID_IDX = self.num_lidar_beams//2
        sparse_idxs = [MID_IDX]

        # Structures
        a = self.lidar_aspect_ratio
        beam_proj = 2*a*np.array([np.cos(beam_angles), np.sin(beam_angles)])
        # Allows us to do intersection math later
        beam_intersections = np.zeros((2, self.num_lidar_beams))

        # Compute the points of intersection along a uniform corridor of given aspect ratio
        box_corners = [(a, 1), (a, -1), (-a, -1), (-a, 1)]
        for idx in range(len(box_corners)):
            x1, y1 = box_corners[idx]
            x2, y2 = box_corners[0] if idx == 3 else box_corners[idx+1]
            for i in range(self.num_lidar_beams):
                x4 = beam_proj[0, i]
                y4 = beam_proj[1, i]

                den = (x1-x2)*(-y4)-(y1-y2)*(-x4)
                if den == 0:
                    continue    # parallel lines

                t = ((x1)*(-y4)-(y1)*(-x4))/den
                u = ((x1)*(y1-y2)-(y1)*(x1-x2))/den

                px = u*x4
                py = u*y4
                if 0 <= t <= 1.0 and 0 <= u <= 1.0:
                    beam_intersections[0, i] = px
                    beam_intersections[1, i] = py

        # Compute the distances for uniform spacing
        dx = np.diff(beam_intersections[0, :])
        dy = np.diff(beam_intersections[1, :])
        dist = np.sqrt(dx**2 + dy**2)
        total_dist = np.sum(dist)
        dist_amt = total_dist/(self.des_lidar_beams-1)
        # rospy.loginfo(f"{dist.shape=}, {total_dist=:.2f}, {dist_amt=:.2f}")

        # Calc half of the evenly-spaced interval first, then the other half
        idx = MID_IDX + 1
        DES_BEAMS2 = self.des_lidar_beams//2 + 1
        acc = 0
        while len(sparse_idxs) <= DES_BEAMS2:
            acc += dist[idx]
            if acc >= dist_amt:
                acc = 0
                sparse_idxs.append(idx-1)
            idx += 1

            if idx == self.num_lidar_beams-1:
                sparse_idxs.append(self.num_lidar_beams-1)
                break

        mirrored_half = []
        for idx in sparse_idxs[1:]:
            new_idx = 2*sparse_idxs[0]-idx
            mirrored_half.insert(0, new_idx)
        sparse_idxs = mirrored_half + sparse_idxs

        self.lidar_sample_idxs = np.array(sparse_idxs)
        self.lidar_theta_lut = beam_angles[self.lidar_sample_idxs]
        self.lidar_theta_lut = self.lidar_theta_lut.astype(np.single)

    def publish_pose_and_tf(self, pose, stamp: Time):
        """ Publish tf and Pose messages for the car. """

        # Avoid re-publishing stamp
        if stamp.seconds_nanoseconds()[0] <= self.last_pub_stamp.seconds_nanoseconds()[0]:
            return

        map_base_link_pos = pose[0:2]
        map_laser_rotation = tft.quaternion_from_euler(0, 0, pose[2])

        header = Utils.make_header("map", stamp)

        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.pose.position.x = map_base_link_pos[0]
        pose_msg.pose.position.y = map_base_link_pos[1]
        pose_msg.pose.orientation.x = map_laser_rotation[0]
        pose_msg.pose.orientation.y = map_laser_rotation[1]
        pose_msg.pose.orientation.z = map_laser_rotation[2]
        pose_msg.pose.orientation.w = map_laser_rotation[3]
        self.pose_pub.publish(pose_msg)

        if self.publish_covariance:
            pose_cov_msg = PoseWithCovarianceStamped()
            pose_cov_msg.pose.pose = pose_msg.pose
            pose_cov_msg.header = header

            covariance = np.zeros((6, 6))
            covariance[0:2, 0:2] = self.cov[0:2, 0:2]   # xy covariances
            covariance[5, 5] = self.cov[2, 2]           # theta variance
            covariance[0:2, 5] = self.cov[0:2, 2]       # xy-theta covariance
            covariance[5, 0:2] = self.cov[0:2, 2]
            pose_cov_msg.pose.covariance = covariance.flatten().tolist()
            self.pose_cov_pub.publish(pose_cov_msg)

        # rospy.loginfo(f"Inferred pose at x: {pose[0]:.2f}, y: {pose[1]:.2f}")

        if self.publish_tf:
            t = TransformStamped()
            t.header = header
            t.child_frame_id = "base_link"
            t.transform.translation.x = map_base_link_pos[0]
            t.transform.translation.y = map_base_link_pos[1]
            t.transform.translation.z = 0.0
            t.transform.rotation.x = map_laser_rotation[0]
            t.transform.rotation.y = map_laser_rotation[1]
            t.transform.rotation.z = map_laser_rotation[2]
            t.transform.rotation.w = map_laser_rotation[3]

            # Publish position/orientation for car_state
            self.pub_tf.sendTransform(t)

        self.last_pub_stamp = self.last_stamp

    def visualise(self):
        '''
        Publish visualization of the particles
        '''
        if not self.use_viz:
            return

        # TODO: future work
        # def publish_particles(particles):
        #     # publish the given particles as a PoseArray object
        #     pa = PoseArray()
        #     pa.header = Utils.make_header("map")
        #     pa.poses = self.particle_utils.particles_to_poses(particles)
        #     self.particle_pub.publish(pa)

        # if self.particle_pub.get_num_connections() > 0:
        #     # publish a downsampled version of the particle distribution to avoid latency
        #     if self.max_particles > self.max_viz_particles:
        #         proposal_indices = np.random.choice(
        #             self.particle_indices, self.max_viz_particles, p=self.weights)
        #         publish_particles(self.particles[proposal_indices, :])
        #     else:
        #         publish_particles(self.particles)
    
    def motion_model(self, proposal_dist, action):
        '''
        The motion model applies the odometry to the particle distribution.

        proposal_dist is a numpy array representing the current belief of the base link.

        action represents dx,dy,dtheta in the lidar frame.

        Uses the MOTION_MODEL parameter (amcl | tum | default) to apply the required transforms.
        '''
        # TODO: debug stuff 
        # if self.DEBUG:
        #     tic = time()

        if self.motion_model_param == 'amcl' or self.motion_model_param == 'tum':
            # Taken from AMCL ROS Package
            a1 = self.alpha_1
            a2 = self.alpha_2
            a3 = self.alpha_3
            a4 = self.alpha_4

            dx = self.curr_pose[0]-self.last_pose[0]
            dy = self.curr_pose[1]-self.last_pose[1]
            dtheta = Utils.angle_diff(self.curr_pose[2], self.last_pose[2])
            d_trans = np.sqrt(dx**2 + dy**2)

            # To prevent drift/instability when we are not moving
            if d_trans < 0.01:
                return

            # Dont calculate rot1 if we are "rotating in place"
            d_rot1 = Utils.angle_diff(np.arctan2(dy, dx), self.last_pose[2]) \
                if d_trans > 0.01 else 0.0

            reverse_offset = 0.0
            reverse_spread = 1.0

            # check if we are reversing
            if len(self.odom_msgs) and self.odom_msgs[-1].twist.twist.linear.x < -0.05:
                reverse_offset = np.pi
                reverse_spread = 1.1
                # Rotate d_rot1 180 degrees
                d_rot1 += np.pi if d_rot1 < -np.pi/2 else -np.pi

            d_rot2 = Utils.angle_diff(dtheta, d_rot1)

            # Enables this to happen in reverse
            d_rot1 = min(np.abs(Utils.angle_diff(d_rot1, 0.0)),
                         np.abs(Utils.angle_diff(d_rot1, np.pi)))
            d_rot2 = min(np.abs(Utils.angle_diff(d_rot2, 0.0)),
                         np.abs(Utils.angle_diff(d_rot2, np.pi)))

            # Debug hooks
            # print(f"dx={dx:.3f} | dy={dy:.3f} | dtheta={np.degrees(dtheta):.3f}")
            # print(f"d_rot1={np.degrees(d_rot1):.3f} | d_rot2={np.degrees(d_rot2):.3f} | d_trans={d_trans:.3f}")

            # TUM model's improvement
            if self.motion_model_param == 'amcl':
                scale_rot1 = a1*d_rot1+a2*d_trans
                scale_rot2 = a1*d_rot2+a2*d_trans
            else:
                # print("d_trans:", d_trans, "v:", self.odom_msgs[-1].twist.twist.linear.x)
                scale_rot1 = a1*d_rot1+a2/(max(d_trans, self.lambda_thresh))
                scale_rot2 = a1*d_rot2+a2/(max(d_trans, self.lambda_thresh))

            scale_trans = a3*d_trans + a4*(d_rot1+d_rot2)

            # If we are reversing, add movement noise
            scale_rot1 *= reverse_spread
            scale_rot2 *= reverse_spread
            scale_trans *= reverse_spread

            d_rot1 += np.random.normal(scale=scale_rot1,
                                       size=self.max_particles)
            # It is more likely that we move forward, so shift the mean of the translation vector.
            # A choice of half a std-deviation is made here.
            d_trans += np.random.normal(loc=scale_trans/2, scale=scale_trans,
                                        size=self.max_particles)
            d_rot2 += np.random.normal(scale=scale_rot2,
                                       size=self.max_particles)

            # ? Future Extension: To add speed-dependent lateral offset to pose
            eff_hdg = proposal_dist[:, 2]+d_rot1+reverse_offset
            proposal_dist[:, 0] += d_trans*np.cos(eff_hdg)
            proposal_dist[:, 1] += d_trans*np.sin(eff_hdg)
            proposal_dist[:, 2] += d_rot1 + d_rot2
        elif self.motion_model_param=='arc':
            dx_, dy_, dtheta_ = action

            # TODO: sanity check if all parameters are transfered correct (groß- und kleinschreibung @Lukas)
            scale_x = max(self.motion_dispersion_arc_x_min, np.abs(dx_) * self.motion_dispersion_arc_x)
            scale_y = np.abs(dy_) * self.motion_dispersion_arc_y
            # If above threshold add noise from x too
            if np.abs(dx_) > self.motion_dispersion_arc_xy_min_x:
                scale_y += (np.abs(dx_)-self.motion_dispersion_arc_xy_min_x)*self.motion_dispersion_arc_xy
            scale_y = min(self.motion_dispersion_arc_y_max, max(self.motion_dispersion_arc_y_min, scale_y))
            scale_th = max(self.motion_dispersion_arc_theta_min, np.abs(dtheta_) * self.motion_dispersion_arc_theta)

            # TODO: code cleaning
            # debugging scales
            # rospy.loginfo(f"{np.abs(dx_)=:.4f} | {scale_x:.4f} | {self.last_odom_msg.twist.twist.linear.x:.4f}")
            # rospy.loginfo(f"{np.abs(dy_)=:.4f} | {scale_y:.4f}")
            # rospy.loginfo(f"{np.abs(dtheta_)=:.4f} | {scale_th:.4f}\n")

            dx = np.random.normal(loc=dx_, scale=scale_x, size=self.max_particles)
            dy = np.random.normal(loc=dy_, scale=scale_y, size=self.max_particles)
            dtheta = np.random.normal(loc=dtheta_, scale=scale_th, size=self.max_particles)

            r = dx/dtheta
            angle = (np.pi/2) * ( np.sign(dtheta)+(dtheta==0) )     # +90 if dtheta is positive
            cx = proposal_dist[:, 0] + r*np.cos(proposal_dist[:, 2] + angle)
            cy = proposal_dist[:, 1] + r*np.sin(proposal_dist[:, 2] + angle)

            psi = np.arctan2(proposal_dist[:, 1]-cy, proposal_dist[:, 0]-cx)

            x_ = cx + r*np.cos(psi + dtheta)
            y_ = cy + r*np.sin(psi + dtheta)
            theta_ = proposal_dist[:, 2] + dtheta

            x_ += dy*np.cos(theta_ + np.pi/2)
            y_ += dy*np.sin(theta_ + np.pi/2)

            proposal_dist[:, 0] = x_
            proposal_dist[:, 1] = y_
            proposal_dist[:, 2] = theta_

        else:
            dx, dy, dtheta = action

            # rotate the action into the coordinate space of each particle
            cosines = np.cos(proposal_dist[:, 2])
            sines = np.sin(proposal_dist[:, 2])

            self.local_deltas[:, 0] = cosines*dx - sines*dy
            self.local_deltas[:, 1] = sines*dx + cosines*dy
            self.local_deltas[:, 2] = dtheta

            proposal_dist[:, :] += self.local_deltas
            proposal_dist[:, 0] += np.random.normal(
                loc=0.0, scale=self.motion_dispersion_x, size=self.max_particles)
            proposal_dist[:, 1] += np.random.normal(
                loc=0.0, scale=self.motion_dispersion_y, size=self.max_particles)
            proposal_dist[:, 2] += np.random.normal(
                loc=0.0, scale=self.motion_dispersion_theta, size=self.max_particles)

        # clamp angles in proposal distribution around (-pi, pi)
        proposal_s = np.sin(proposal_dist[:,2])
        proposal_c = np.cos(proposal_dist[:,2])
        proposal_dist[:,2] = np.arctan2(proposal_s, proposal_c)

        # TODO: debug remove
        # if self.DEBUG:
        #     self.motion_model_time_ms.append((time()-tic)*1000)

        # TODO: ? Future Extension: resample if the particle goes out of track?

    def sensor_model(self, proposal_dist, obs, weights):
        '''
        This function computes a probablistic weight for each particle in the proposal distribution.
        These weights represent how probable each proposed (x,y,theta) pose is given the measured
        ranges from the lidar scanner.

        There are 4 different variants using various features of RangeLibc for demonstration purposes.
        - VAR_REPEAT_ANGLES_EVAL_SENSOR is the most stable, and is very fast.
        - VAR_NO_EVAL_SENSOR_MODEL directly indexes the precomputed sensor model. This is slow
                                   but it demonstrates what self.range_method.eval_sensor_model does
        - VAR_RADIAL_CDDT_OPTIMIZATIONS is only compatible with CDDT or PCDDT, it implments the radial
                                        optimizations to CDDT which simultaneously performs ray casting
                                        in two directions, reducing the amount of work by roughly a third
        '''
        
        # TODO: remove debug
        # if self.DEBUG:
        #     tic = time()
            
        num_rays = self.des_lidar_beams
        # only allocate buffers once to avoid slowness
        if self.first_sensor_update:
            if self.rangelib_variant == RangeLibVariant.VAR_NO_EVAL_SENSOR_MODEL or \
               self.rangelib_variant == RangeLibVariant.VAR_CALC_RANGE_MANY_EVAL_SENSOR:

                self.get_logger().warn("""In these modes, the proposal_dist is not transformed from the base_links to the laser frame.
                              Performance of PF will be worse. Try using another variant instead, as they offer better speed anyway!""")

                self.queries = np.zeros(
                    (num_rays*self.max_particles, 3), dtype=np.float32)
            else:
                self.queries = np.zeros(
                    (self.max_particles, 3), dtype=np.float32)

            self.ranges = np.zeros(
                num_rays*self.max_particles, dtype=np.float32)
            self.tiled_angles = np.tile(
                self.lidar_theta_lut, self.max_particles)
            self.first_sensor_update = False

        # transform particles into the laser frame
        proposal_s = np.sin(proposal_dist[:,2])
        proposal_c = np.cos(proposal_dist[:,2])
        rot = np.array([[proposal_c, -proposal_s],
                        [proposal_s, proposal_c]]).transpose(2,0,1)   # (N,2,2)
        laser_offset_2d = self.laser_base_link_offset[:2]
        res = ( rot @ laser_offset_2d[np.newaxis, :, np.newaxis] ).reshape(self.max_particles, 2)

        self.queries[:, :] = proposal_dist
        self.queries[:, :2] += res

        # ! THIS assumes constant angle between scans but we don't do this.
        if self.rangelib_variant == RangeLibVariant.VAR_RADIAL_CDDT_OPTIMIZATIONS:
            if "cddt" in self.range_method:
                # self.queries[:, :] = proposal_dist[:, :]
                self.range_method.calc_range_many_radial_optimized(
                    num_rays, self.downsampled_angles[0], self.downsampled_angles[-1], self.queries, self.ranges)

                # evaluate the sensor model
                self.range_method.eval_sensor_model(
                    obs, self.ranges, self.weights, num_rays, self.max_particles)
                # apply the squash factor
                self.weights = np.power(self.weights, self.inv_squash_factor)
            else:
                raise ValueError(
                    "Cannot use radial optimizations with non-CDDT based methods, use rangelib_variant 2")
        elif self.rangelib_variant == RangeLibVariant.VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT:
            # self.queries[:, :] = proposal_dist[:, :]
            self.range_method.calc_range_repeat_angles_eval_sensor_model(
                self.queries, self.lidar_theta_lut, obs, self.weights)
            np.power(self.weights, self.inv_squash_factor, self.weights)
        elif self.rangelib_variant == RangeLibVariant.VAR_REPEAT_ANGLES_EVAL_SENSOR:
            # this version demonstrates what this would look like with coordinate space conversion pushed to rangelib
            # self.queries[:, :] = proposal_dist[:, :]
            self.range_method.calc_range_repeat_angles(
                self.queries, self.lidar_theta_lut, self.ranges)

            # evaluate the sensor model on the GPU
            self.range_method.eval_sensor_model(
                obs, self.ranges, self.weights, num_rays, self.max_particles)
            np.power(self.weights, self.inv_squash_factor, self.weights)
        elif self.rangelib_variant == RangeLibVariant.VAR_CALC_RANGE_MANY_EVAL_SENSOR:
            # this version demonstrates what this would look like with coordinate space conversion pushed to rangelib
            # this part is inefficient since it requires a lot of effort to construct this redundant array
            self.queries[:, 0] = np.repeat(proposal_dist[:, 0], num_rays)
            self.queries[:, 1] = np.repeat(proposal_dist[:, 1], num_rays)
            self.queries[:, 2] = np.repeat(proposal_dist[:, 2], num_rays)
            self.queries[:, 2] += self.tiled_angles

            self.range_method.calc_range_many(self.queries, self.ranges)

            # evaluate the sensor model on the GPU
            self.range_method.eval_sensor_model(
                obs, self.ranges, self.weights, num_rays, self.max_particles)
            np.power(self.weights, self.inv_squash_factor, self.weights)
        elif self.rangelib_variant == RangeLibVariant.VAR_NO_EVAL_SENSOR_MODEL:
            # this version directly uses the sensor model in Python, at a significant computational cost
            self.queries[:, 0] = np.repeat(proposal_dist[:, 0], num_rays)
            self.queries[:, 1] = np.repeat(proposal_dist[:, 1], num_rays)
            self.queries[:, 2] = np.repeat(proposal_dist[:, 2], num_rays)
            self.queries[:, 2] += self.tiled_angles

            # compute the ranges for all the particles in a single functon call
            self.range_method.calc_range_many(self.queries, self.ranges)

            # resolve the sensor model by discretizing and indexing into the precomputed table
            obs /= float(self.map_info.resolution)
            ranges = self.ranges / float(self.map_info.resolution)
            obs[obs > self.max_range_px] = self.max_range_px
            ranges[ranges > self.max_range_px] = self.max_range_px

            intobs = np.rint(obs).astype(np.uint16)
            intrng = np.rint(ranges).astype(np.uint16)

            # compute the weight for each particle
            for i in range(self.max_particles):
                # TODO: changed np.product() to np.prod
                # weight = np.product(
                #     self.sensor_model_table[intobs, intrng[i*num_rays:(i+1)*num_rays]])
                weight = np.prod(
                    self.sensor_model_table[intobs, intrng[i*num_rays:(i+1)*num_rays]])
                weight = np.power(weight, self.inv_squash_factor)
                weights[i] = weight
        else:
            raise ValueError(
                f"Please set rangelib_variant param to 0-4. Current value: {self.rangelib_variant}")
        
        # TODO: remove debug
        # if self.DEBUG:
        #     self.sensor_model_time_ms.append((time()-tic)*1000)
    
    def MCL(self, odom_data, observations):
        '''
        Performs one step of Monte Carlo Localization.
            1. resample particle distribution to form the proposal distribution
            2. apply the motion model
            3. apply the sensor model
            4. normalize particle weights
        '''
        # draw the proposal distribution from the old particles
        proposal_indices = np.random.choice(
            self.particle_indices, self.max_particles, p=self.weights)
        proposal_distribution = self.particles[proposal_indices, :]

        # compute the motion model to update the proposal distribution
        self.motion_model(proposal_distribution, odom_data)

        # compute the sensor model
        self.sensor_model(proposal_distribution, observations, self.weights)

        # check for permissible region and downscale weight if a particle goes out of track
        # TODO: ? Future Extension: express map in 'world' coordinates to save on this optimization?
        particles_in_map = np.copy(proposal_distribution)
        Utils.world_to_map(particles_in_map, self.map_info)
        limit = self.permissible_region.shape
        particles_in_map = np.clip(particles_in_map[:, 0:2].astype('int'),[0, 0], [limit[1]-1, limit[0]-1])
        valid_particles = self.permissible_region[particles_in_map[:,1], particles_in_map[:, 0]]
        self.weights = np.where(valid_particles, self.weights, 0.01*self.weights)

        # normalize importance weights
        weight_sum = np.sum(self.weights)

        # Empirically tuned term
        # TODO: code cleaning 
        if False:
        # if weight_sum < 1e-16:
            # ? Future Extension: Send messages somewhere to alert a safety controller
            rospy.logerr("Particle depletion occured!")
            # First release the state lock to effect change in global init
            self.state_lock.release()
            self.initialize_global()
            self.state_lock.acquire()
        else:
            self.weights /= weight_sum

        # save the particles
        self.particles = proposal_distribution

        # Compute particle covariance about the inferred pose
        if self.inferred_pose is not None and self.publish_covariance:
            spread = self.particles - self.inferred_pose    #(N,3)
            # Distance in theta calculation to be (-pi,pi)
            spread[spread[:,2]> np.pi, 2] -= 2*np.pi
            spread[spread[:,2]<-np.pi, 2] += 2*np.pi

            spread = spread[:, :, np.newaxis]                               # (N,3,1)
            inner_prod = spread @ spread.transpose(0, 2, 1)                 # (N,3,1) @ (N,1,3) = (N,3,3)
            res = self.weights[:, np.newaxis, np.newaxis] * inner_prod      # (N,1,1)*(N,3,3) = (N,3,3)
            self.cov = np.sum(res, axis=0)   
    
    def update(self):
        '''
        Apply the MCL function to update particle filter state.

        Ensures the state is correctly initialized, and acquires the state lock before proceeding.
        '''
        if not (self.lidar_initialized and self.odom_initialized and self.map_initialized):
            return

        if self.state_lock.locked():
            self.get_logger().info("Concurrency error avoided")
            return

        self.state_lock.acquire()

        observation = np.copy(self.downsampled_ranges).astype(np.float32)

        # run the MCL update algorithm
        self.MCL(self.odometry_data, observation)
        self.odometry_data = np.zeros(3)

        # compute the expected value of the robot pose
        inferred_x = np.sum(self.particles[:,0] * self.weights)
        inferred_y = np.sum(self.particles[:,1] * self.weights)
        inferred_s = np.sum(np.sin(self.particles[:,2]) * self.weights)
        inferred_c = np.sum(np.cos(self.particles[:,2]) * self.weights)

        self.inferred_pose = np.array(
            ( inferred_x, inferred_y , np.arctan2(inferred_s, inferred_c) )
        )

        self.state_lock.release()

        # publish transformation frame based on inferred pose
        self.publish_pose_and_tf(self.inferred_pose, self.last_stamp)
        self.visualise()
    
    def localization_loop(self):
        #  TOOD: remove debug
        # if self.DEBUG:
        #     self.profiler.enable()
        #     tic = time()

        # Assert odom has been initialised / there are things to process
        if len(self.odom_msgs) == 0:
            return

        # Get the furthest-back message in the buffer
        msg: Odometry = self.odom_msgs[0]

        # Quantify gap in message pubs (this was solved with the tcp_nodelay setting)
        # msg_time = msg.header.stamp.to_sec()
        # now_time = rospy.Time.now().to_sec()
        # rospy.loginfo(f"Time gap to present: {(now_time-msg_time)*1000:.2f}ms | Buffer Size: {len(self.odom_msgs)}")

        position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y])

        orientation = Utils.quaternion_to_angle(msg.pose.pose.orientation)
        self.curr_pose = np.array([position[0], position[1], orientation])

        self.update()   # Update based on curr_pose and last_pose

        if isinstance(self.last_pose, np.ndarray):
            # changes in x,y,theta in local coordinate system of the car
            rot = Utils.rotation_matrix(-self.last_pose[2])
            delta = np.array(
                [position - self.last_pose[0:2]]).transpose()
            local_delta = (rot*delta).transpose()
            self.odometry_data = np.array(
                [local_delta[0, 0], local_delta[0, 1], orientation - self.last_pose[2]])

            self.odom_initialized = True
        else:
            self.get_logger().info("PF2...Received first Odometry message")

        # self.last_stamp = msg.header.stamp
        self.last_stamp = self.get_clock().now()
        self.last_pose = self.curr_pose
        self.odom_msgs.popleft()

        self.rate.sleep()

        # TODO: remove debug
        # if self.DEBUG:
        #     self.profiler.disable()
        #     self.overall_time_ms.append((time()-tic)*1000)

        #     if (self.itr % 30) == 0:
        #         s_mean = np.mean(self.sensor_model_time_ms)
        #         s_std = np.std(self.sensor_model_time_ms)
        #         m_mean = np.mean(self.motion_model_time_ms)
        #         m_std = np.std(self.motion_model_time_ms)
        #         o_mean = np.mean(self.overall_time_ms)
        #         o_std = np.std(self.overall_time_ms)

        #         rospy.loginfo(
        #             f"Sensor Model: {s_mean:4.2f}ms std:{s_std:4.2f}ms | Motion Model: {m_mean:4.2f}ms std:{m_std:4.2f}ms | Overall: {o_mean:4.2f}ms std:{o_std:4.2f}ms")

        #     if (self.itr % 500) == 0:
        #         stats = pstats.Stats(self.profiler)
        #         stats.sort_stats(pstats.SortKey.TIME)
        #         # look for this in ~/.ros
        #         stats.dump_stats(filename="pf2_stats.prof")
        #         rospy.logwarn("PF2 Dumping profiling stats to file.")

        #     self.itr += 1


def main(args=None):
    rclpy.init(args=args)
    particle_filter_node = ParticleFilter()
    
    try:
        rclpy.spin(particle_filter_node)
    except KeyboardInterrupt:
        pass
    finally:
        particle_filter_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
import os
import numpy as np
import wandb
import uuid
import traceback
import logging
import math
from datetime import datetime
import plotly.graph_objects as go
from pathlib import Path
import time
import math

# Set WANDB_MODE to 'disabled' to prevent local file creation
os.environ["WANDB_MODE"] = "disabled"

# Configure logging
logger = logging.getLogger(__name__)

# def dh_transform(a, alpha, d, theta):
#     """
#     计算单个DH变换矩阵
#     """
#     return np.array([
#         [math.cos(theta), -math.sin(theta)*math.cos(alpha), math.sin(theta)*math.sin(alpha), a*math.cos(theta)],
#         [math.sin(theta), math.cos(theta)*math.cos(alpha), -math.cos(theta)*math.sin(alpha), a*math.sin(theta)],
#         [0, math.sin(alpha), math.cos(alpha), d],
#         [0, 0, 0, 1]
#     ])

# def forward_kinematics(joint_angles):
#     """
#     使用DH参数计算末端执行器的位置
#     """
#     # 初始化变换矩阵为单位矩阵
#     T = np.eye(4)
#     armConfig = {
#         "L_BASE": 0.035,       # 基座宽度
#         "D_BASE": 0.109,       # 基座高度
#         "L_ARM": 0.146,        # 大臂长度
#         "D_ELBOW": 0.052,      # 肘部偏移
#         "L_FOREARM": 0.115,    # 前臂长度
#         "L_WRIST": 0.072       # 腕部长度
#     }
#     DH_matrix = [
#         [0.0, armConfig["L_BASE"], armConfig["D_BASE"], -math.pi / 2],  # 第一关节
#         [-math.pi / 2, 0.0, armConfig["L_ARM"], 0.0],  # 第二关节
#         [math.pi / 2, armConfig["D_ELBOW"], 0.0, math.pi / 2],  # 第三关节
#         [0.0, armConfig["L_FOREARM"], 0.0, -math.pi / 2],  # 第四关节
#         [0.0, 0.0, 0.0, math.pi / 2],  # 第五关节
#         [0.0, armConfig["L_WRIST"], 0.0, 0.0]  # 第六关节
#     ]
#     # 计算每个关节的变换矩阵并相乘
#     for i in range(6):
#         a = DH_matrix[i][1]  # 连杆长度
#         alpha = DH_matrix[i][0]  # 连杆扭转角
#         d = DH_matrix[i][2]  # 连杆偏移量
#         theta = joint_angles[i]  # 关节角度

#         Ti = dh_transform(a, alpha, d, theta)
#         T = np.dot(T, Ti)  # 乘以变换矩阵

#     # 末端执行器位置是变换矩阵的最后一列
#     end_effector_position = T[:3, 3]

#     return [round(end_effector_position[0], 4), round(end_effector_position[1], 4), round(end_effector_position[2], 4)]

def forward_kinematics(joint_angles):
    """
    Calculate end effector position (x, y, z) based on joint angles
    
    Note: This is a simplified model, adjust DH parameters for specific robot models
    
    Args:
        joint_angles: List of joint angles [j1, j2, j3, j4, j5, j6, gripper]
    
    Returns:
        List containing end position [x, y, z]
    """
    # Convert degrees to radians
    angles_rad = [math.radians(angle) for angle in joint_angles[:6]]  # Only use 6 joint angles, excluding gripper
    
    # DH parameters (adjust according to specific robot model)
    # These are estimated parameters, use exact parameters from robot manual in actual applications
    d1 = 0.109  # Base to first joint height (m)
    a2 = 0.146  # First joint to second joint length (m)
    a3 = 0 # Second joint to third joint length (m)
    d5 = 0.072  # Wrist joint offset (m)
    d6 = 0 # End effector length (m)
    
    # Joint 1 - Base rotation
    j1 = angles_rad[0]
    # Joint 2 - Shoulder joint
    j2 = angles_rad[1]
    # Joint 3 - Elbow joint
    j3 = angles_rad[2]
    # Joints 4, 5, 6 - Wrist joint
    j4 = angles_rad[3]
    j5 = angles_rad[4]
    j6 = angles_rad[5]
    
    # Calculate position based on joint angles
    # This is a simplified calculation, use complete DH transformation matrix in practice
    
    # Shoulder position
    shoulder_x = 0
    shoulder_y = 0
    shoulder_z = d1
    
    # Elbow position
    elbow_x = a2 * math.cos(j1) * math.cos(j2)
    elbow_y = a2 * math.sin(j1) * math.cos(j2)
    elbow_z = shoulder_z + a2 * math.sin(j2)
    
    # Wrist position
    wrist_x = elbow_x + a3 * math.cos(j1) * math.cos(j2 + j3)
    wrist_y = elbow_y + a3 * math.sin(j1) * math.cos(j2 + j3)
    wrist_z = elbow_z + a3 * math.sin(j2 + j3)
    
    # End effector position
    end_x = wrist_x + (d5 + d6) * (math.cos(j1) * math.cos(j2 + j3) * math.cos(j4) * math.cos(j5) * math.cos(j6) -
                                  math.cos(j1) * math.sin(j2 + j3) * math.sin(j5) * math.cos(j6) -
                                  math.cos(j1) * math.cos(j2 + j3) * math.sin(j4) * math.sin(j6))
    
    end_y = wrist_y + (d5 + d6) * (math.sin(j1) * math.cos(j2 + j3) * math.cos(j4) * math.cos(j5) * math.cos(j6) -
                                  math.sin(j1) * math.sin(j2 + j3) * math.sin(j5) * math.cos(j6) -
                                  math.sin(j1) * math.cos(j2 + j3) * math.sin(j4) * math.sin(j6))
    
    end_z = wrist_z + (d5 + d6) * (math.sin(j2 + j3) * math.cos(j4) * math.cos(j5) * math.cos(j6) +
                                  math.cos(j2 + j3) * math.sin(j5) * math.cos(j6) -
                                  math.sin(j2 + j3) * math.sin(j4) * math.sin(j6))
    
    # Return end effector position (meters)
    return [round(end_x, 4), round(end_y, 4), round(end_z, 4)]

def ensure_wandb_login():
    """Ensure WandB is logged in, with helpful error messages if not"""
    try:
        # Check if already logged in
        api = wandb.Api()
        user = api.viewer()
        logger.info(f"Logged in to WandB as: {user.get('username', 'unknown')}")
        logger.info(f"Available WandB teams: {', '.join([e['name'] for e in user.get('teams', [])])}")
        return True
    except Exception as e:
        logger.warning(f"WandB login check failed: {e}")
        logger.warning("Not logged in to Weights & Biases.")
        logger.warning("To log trajectories to WandB, please either:")
        logger.warning("1. Run 'wandb login' in your terminal before running this script")
        logger.warning("2. Pass --wandb_api_key YOUR_API_KEY when running the script")
        logger.warning("3. Set the WANDB_API_KEY environment variable")
        logger.warning("Will continue with local trajectory logging only.")
        return False


def force_create_wandb_project(project_name, entity=None):
    """
    Explicitly create a WandB project if it doesn't exist
    This helps resolve issues where projects aren't automatically created
    """
    try:
        api = wandb.Api()
        
        # Determine entity to use
        if entity:
            target_entity = entity
        else:
            # Use current user if no entity specified
            user = api.viewer()
            target_entity = user.get('username')
            
        logger.info(f"Ensuring WandB project exists: {target_entity}/{project_name}")
        
        # Try to get the project - if it fails with 404, create it
        try:
            api.project(target_entity, project_name)
            logger.info(f"WandB project already exists: {target_entity}/{project_name}")
        except Exception as e:
            if "404" in str(e):
                # Create new project
                logger.info(f"Creating new WandB project: {target_entity}/{project_name}")
                api.create_project(project_name, entity=target_entity)
                logger.info(f"Successfully created WandB project: {target_entity}/{project_name}")
            else:
                raise e
                
        return True
    except Exception as e:
        logger.error(f"Failed to create WandB project: {e}")
        return False


def disable_wandb_sync():
    """
    禁用WandB的W&B进程同步，这可能会导致某些环境中的问题
    """
    try:
        os.environ["WANDB_START_METHOD"] = "thread"
        os.environ["WANDB_DISABLE_SERVICE"] = "true"
        os.environ["WANDB_SILENT"] = "true"
        os.environ["WANDB_DISABLE_SYMLINKS"] = "true"
        logger.info("Disabled WandB service sync (using thread mode)")
    except Exception as e:
        logger.warning(f"Failed to set WandB environment variables: {e}")

# Call this function immediately to disable wandb sync
disable_wandb_sync()

class TrajectoryVisualizer:
    def __init__(self, project_name="robot-trajectory", experiment_name=None, entity=None, wandb_api_key=None):
        """
        Initialize trajectory visualizer with WandB integration
        
        Args:
            project_name: WandB project name
            experiment_name: Experiment name, defaults to timestamp
            entity: WandB entity (username or team name)
            wandb_api_key: WandB API key to use for authentication
        """
        # Ensure WANDB_MODE is set to disabled to prevent local file storage
        os.environ["WANDB_MODE"] = "disabled"
        # Also disable symlinks and file sync which create local files
        os.environ["WANDB_DISABLE_SYMLINKS"] = "true"
        os.environ["WANDB_SILENT"] = "true"
        
        # If no experiment name provided, create unique identifier with current time
        if experiment_name is None:
            experiment_name = f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.positions = []
        self.pred_positions = []
        self.timestamps = []
        
        # Set WandB API key if provided
        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key
            logger.info("Using provided WandB API key for trajectory visualization")
        
        # Initialize WandB - handle authentication issues gracefully
        try:
            # Configure WandB with explicit parameters
            wandb_config = {
                "project": project_name,
                "name": experiment_name,
                "reinit": True,  # Allow reinitializing the run if needed
            }
            
            # Add entity if provided
            if entity:
                wandb_config["entity"] = entity
                logger.info(f"Using WandB entity: {entity}")
            
            # Force create project before init
            if not force_create_wandb_project(project_name, entity):
                logger.warning(f"未能强制创建或确认WandB项目: {entity}/{project_name}。 初始化可能会失败。")
            
            # Try to initialize WandB with config
            wandb.init(**wandb_config)
            
            # Log basic info
            wandb.config.update({
                "experiment_name": experiment_name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "system": "robot-arm-trajectory"
            })
            
            self.wandb_initialized = True
            wandb_url = wandb.run.get_url()
            self.wandb_url = wandb_url
            
            # Print prominent message with URL
            logger.info(f"Successfully initialized WandB project: {project_name}, experiment: {experiment_name}")
            logger.info("=" * 80)
            logger.info(f"WandB Visualization URL: {wandb_url}")
            logger.info("Open this URL in your browser to view real-time trajectory visualization")
            logger.info("=" * 80)
            
            # Also print to stdout for direct visibility
            print("\n" + "=" * 80)
            print(f"🔗 WandB Visualization URL: {wandb_url}")
            print("Open this URL in your browser to view real-time trajectory visualization")
            print("=" * 80 + "\n")
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}", exc_info=True)
            logger.warning("Will continue without WandB logging. Use 'wandb login' to authenticate if needed.")
            self.wandb_initialized = False
            self.wandb_url = None
    
    def add_position(self, timestamp, position, pred_position=None):
        """
        Add position point to trajectory
        
        Args:
            timestamp: Timestamp
            position: Current position [x, y, z]
            pred_position: Predicted position [x, y, z], optional
        """
        self.timestamps.append(timestamp)
        self.positions.append(position)
        if pred_position is not None:
            self.pred_positions.append(pred_position)
    
    def create_3d_trajectory(self):
        """
        Create 3D trajectory visualization chart
        
        Returns:
            Plotly figure object
        """
        # Extract coordinates
        x = [pos[0] for pos in self.positions]
        y = [pos[1] for pos in self.positions]
        z = [pos[2] for pos in self.positions]
        
        # Create actual trajectory
        trace_actual = go.Scatter3d(
            x=z, y=y, z=x,  # Swap x and z axes
            mode='lines+markers',
            name='Actual Position',
            marker=dict(
                size=4,
                color='blue',
            ),
            line=dict(
                color='blue',
                width=2
            )
        )
        
        data = [trace_actual]
        
        # If there are predicted positions, add prediction trajectory
        if self.pred_positions:
            pred_x = [pos[0] for pos in self.pred_positions]
            pred_y = [pos[1] for pos in self.pred_positions]
            pred_z = [pos[2] for pos in self.pred_positions]
            
            trace_pred = go.Scatter3d(
                x=pred_z, y=pred_y, z=pred_x,  # Swap x and z axes
                mode='lines+markers',
                name='Predicted Position',
                marker=dict(
                    size=4,
                    color='red',
                ),
                line=dict(
                    color='red',
                    width=2
                )
            )
            
            data.append(trace_pred)
        
        # Create chart
        fig = go.Figure(data=data)
        
        # Set layout
        fig.update_layout(
            title='Robot Arm End Effector Trajectory',
            scene=dict(
                xaxis_title='Z Position (m)',  # Swap axis labels
                yaxis_title='Y Position (m)',
                zaxis_title='X Position (m)',  # Swap axis labels
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(x=0, y=0),
            template="plotly_dark"
        )
        
        return fig
    
    def log_trajectory(self):
        """Generate trajectory chart and upload to WandB"""
        if not self.positions:
            logger.warning("No trajectory data to visualize")
            return None # Return None if no data
            
        # Create 3D trajectory chart
        fig = self.create_3d_trajectory()
        
        # Upload to WandB only if initialized
        if self.wandb_initialized:
            try:
                # Only add custom 3D point cloud to WandB
                self._log_3d_point_cloud()
                
                # Periodically remind the user about the visualization URL
                if len(self.positions) % 100 == 0:
                    print(f"\n🔄 New data point added: {len(self.positions)} points in trajectory")
                    print(f"🔗 View live visualization at: {self.wandb_url}\n")
            except Exception as e:
                logger.error(f"Failed to log to WandB: {e}")
                logger.error(traceback.format_exc())
        else:
            logger.info(f"Trajectory visualization ready, total {len(self.positions)} points (WandB not initialized)")
            
        return fig # Return the figure object
    
    def _log_3d_point_cloud(self):
        """Log 3D point cloud data to WandB"""
        try:
            # Convert positions to numpy array for easier handling
            positions = np.array(self.positions)
            if len(positions) == 0:
                return
                
            # Create point cloud data
            data = np.zeros(len(positions), dtype=[
                ('x', float), ('y', float), ('z', float), 
                ('color', np.uint8, 3), ('timestamp', float)
            ])
            
            # Fill the point cloud data
            data['x'] = positions[:, 2]  # Z becomes X
            data['y'] = positions[:, 1]  # Y stays as Y
            data['z'] = positions[:, 0]  # X becomes Z
            
            # Generate color gradient based on time (blue to red)
            if len(self.timestamps) > 0:
                timestamps = np.array(self.timestamps)
                # Normalize timestamps to 0-1 range
                t_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min()) if len(timestamps) > 1 else np.zeros_like(timestamps)
                
                # Blue (start) to red (end) gradient
                colors = np.zeros((len(t_norm), 3), dtype=np.uint8)
                colors[:, 0] = (255 * t_norm).astype(np.uint8)  # R increases with time
                colors[:, 2] = (255 * (1 - t_norm)).astype(np.uint8)  # B decreases with time
                data['color'] = colors
                data['timestamp'] = timestamps
            
            # Log to WandB as a 3D scatter plot
            point_cloud = wandb.Object3D({
                'type': 'scatter',
                'points': data,
                'box': True,
                'vectors': []
            })
            
            # Create predicted trajectory point cloud if available
            pred_point_cloud = None
            if hasattr(self, 'pred_positions') and self.pred_positions:
                pred_positions = np.array(self.pred_positions)
                if len(pred_positions) > 0:
                    pred_data = np.zeros(len(pred_positions), dtype=[
                        ('x', float), ('y', float), ('z', float), 
                        ('color', np.uint8, 3)
                    ])
                    
                    pred_data['x'] = pred_positions[:, 2]  # Z becomes X
                    pred_data['y'] = pred_positions[:, 1]  # Y stays as Y
                    pred_data['z'] = pred_positions[:, 0]  # X becomes Z
                    
                    # Use green color for predictions
                    pred_data['color'] = np.tile([0, 255, 0], (len(pred_positions), 1))
                    
                    pred_point_cloud = wandb.Object3D({
                        'type': 'scatter',
                        'points': pred_data,
                        'box': True,
                        'vectors': []
                    })
            
            # Log both point clouds
            log_data = {"Trajectory_3D": point_cloud}
            if pred_point_cloud:
                log_data["Predicted_Trajectory_3D"] = pred_point_cloud
                
            wandb.log(log_data)
            
        except Exception as e:
            logger.error(f"Failed to log 3D point cloud: {e}")
            logger.error(traceback.format_exc())
    
    def close(self):
        """End WandB recording"""
        if self.wandb_initialized:
            try:
                wandb.finish()
                logger.info("WandB session closed")
            except Exception as e:
                logger.error(f"Error closing WandB session: {e}")


class DataLogger:
    def __init__(self, log_dir, forward_kinematics_fn=None, use_wandb=False, wandb_project="robot-arm-trajectory", wandb_entity=None, wandb_api_key=None):
        """
        Initialize data logger with optional WandB integration
        
        Args:
            log_dir: Directory to save logs
            forward_kinematics_fn: Function to calculate forward kinematics (optional, default: internal implementation)
            use_wandb: Whether to use WandB visualization (default: False)
            wandb_project: WandB project name
            wandb_entity: WandB entity (username/team)
            wandb_api_key: API key for WandB authentication
        """
        # Check for WANDB_MODE environment variable - if set to disabled, override use_wandb
        if os.environ.get("WANDB_MODE") == "disabled":
            use_wandb = False
            logger.info("WandB disabled via environment variable WANDB_MODE=disabled")
            
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # ---> MODIFIED: Use timestamp for folder name <---
        # Generate a unique timestamp-based folder name
        self.run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.current_folder_path = self.log_dir / self.run_timestamp
        self.current_folder_path.mkdir(exist_ok=True)
        # ---> END MODIFIED <---
        
        # Use provided function or default to internal implementation
        self.forward_kinematics = forward_kinematics_fn if forward_kinematics_fn is not None else forward_kinematics
        
        # Create image subfolders
        self.wrist_image_folder = self.current_folder_path / "image_wrist"
        self.wrist_image_folder.mkdir(exist_ok=True)
        
        self.head_image_folder = self.current_folder_path / "image_head"
        self.head_image_folder.mkdir(exist_ok=True)
        
        # Initialize state data list
        self.state_data = []
        
        # State JSON file path
        self.state_json_path = self.current_folder_path / "state.json"
        
        # Store WandB parameters
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_api_key = wandb_api_key
        
        # Initialize WandB trajectory visualizer
        if self.use_wandb:
            # ---> MODIFIED: Use timestamp for experiment name <---
            # Ensure WandB experiment name matches the folder name
            wandb_experiment_name = f"trajectory_{self.run_timestamp}"
            self.trajectory_visualizer = TrajectoryVisualizer(
                project_name=self.wandb_project,
                experiment_name=wandb_experiment_name, # Use timestamp name
                entity=self.wandb_entity,
                wandb_api_key=self.wandb_api_key
            )
            # ---> END MODIFIED <---
        else:
            self.trajectory_visualizer = None
            
        logger.info(f"Created log folder structure at: {self.current_folder_path}")
        if self.use_wandb and self.trajectory_visualizer and self.trajectory_visualizer.wandb_initialized:
             logger.info(f"WandB logging enabled: Run Name={wandb_experiment_name}, Project={self.wandb_project}, Entity={self.wandb_entity or 'default'}")
             logger.info(f"WandB URL: {self.trajectory_visualizer.wandb_url}")
        elif self.use_wandb:
            logger.warning("WandB logging was requested but failed to initialize.")
    
    def finalize(self):
        """Finalize recording and cleanup resources"""
        # Upload final trajectory visualization to WandB
        if self.use_wandb and self.trajectory_visualizer:
            try:
                if self.trajectory_visualizer.wandb_initialized:
                    # Log the final trajectory state if there are points
                    if self.trajectory_visualizer.positions:
                         self.trajectory_visualizer.log_trajectory()
                    
                    # 最后一次打印可视化链接
                    print("\n" + "=" * 80)
                    print(f"🎉 数据采集完成! 共 {len(self.state_data)} 条记录")
                    if self.trajectory_visualizer.wandb_url:
                        print(f"🔗 完整可视化链接: {self.trajectory_visualizer.wandb_url}")
                        print("轨迹在WandB中永久保存，可以随时查看")
                    else:
                        print("WandB 未成功初始化，无法提供可视化链接。")
                    print("=" * 80 + "\n")
                    
                    self.trajectory_visualizer.close()
            except Exception as e:
                logger.error(f"Error finalizing WandB visualization: {e}")
            
        logger.info(f"Data recording completed, {len(self.state_data)} records total")
    
    def save_data(self, image_wrist, image_head, state, prediction=None, inference_time_ms=None, cv2=None):
        """
        Save image and state data to the current folder structure
        
        Args:
            image_wrist: Wrist camera image (numpy array, assumed RGB)
            image_head: Head camera image (numpy array, assumed RGB)
            state: Current state (joint angles)
            prediction: Predicted state (optional)
            inference_time_ms: Inference time in milliseconds (optional)
            cv2: OpenCV module (pass in to avoid circular imports)
        """
        # Generate timestamp for unique filenames
        timestamp = int(time.time() * 1000)  # milliseconds since epoch
        
        # Import locally to avoid circular imports
        if cv2 is None:
            import cv2 as cv2_local
            cv2 = cv2_local
        
        # Prepare wrist image (convert to BGR for saving with cv2)
        wrist_img_bgr = None
        wrist_img_rgb = None # Keep RGB for logging
        if image_wrist is not None:
            # Ensure it's uint8
            img_uint8 = (image_wrist * 255).astype(np.uint8)
            wrist_img_rgb = img_uint8 # Keep RGB for potential wandb logging
            wrist_img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            wrist_img_path = self.wrist_image_folder / f"{timestamp}.jpg"
            cv2.imwrite(str(wrist_img_path), wrist_img_bgr)
        
        # Prepare head image (convert to BGR for saving with cv2)
        head_img_bgr = None
        head_img_rgb = None # Keep RGB for logging
        if image_head is not None:
            # Ensure it's uint8
            img_uint8 = (image_head * 255).astype(np.uint8)
            head_img_rgb = img_uint8 # Keep RGB for potential wandb logging
            head_img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            head_img_path = self.head_image_folder / f"{timestamp}.jpg"
            cv2.imwrite(str(head_img_path), head_img_bgr)
        
        # Calculate end effector position from state
        end_position = self.forward_kinematics(state)
        
        # Create data entry with timestamp, state and end effector position
        entry = {
            "timestamp": timestamp,
            "state": state,
            "end_position": end_position  # Add end position [x, y, z]
        }
        
        # Predicted position
        pred_end_position = None
        
        # Add prediction data if available - ensure it's properly converted to list
        if prediction is not None:
            # Make sure prediction is a list (not numpy array) for JSON serialization
            if hasattr(prediction, 'tolist'):
                entry["prediction"] = prediction.tolist()
            else:
                entry["prediction"] = list(prediction)
                
            # Also calculate predicted end effector position
            pred_end_position = self.forward_kinematics(prediction)
            entry["pred_end_position"] = pred_end_position
        
        # Add inference time if available
        if inference_time_ms is not None:
            entry["inference_time_ms"] = inference_time_ms
        
        # Add to state data list
        self.state_data.append(entry)
        
        # Save updated state data to JSON file
        import json
        with open(self.state_json_path, "w") as f:
            json.dump(self.state_data, f, indent=2)

        # Add to WandB trajectory visualization
        if self.use_wandb and self.trajectory_visualizer:
            self.trajectory_visualizer.add_position(timestamp, end_position, pred_end_position)

            # Update WandB visualization in real-time for every data point
            try:
                # Check if WandB is initialized before logging
                if self.trajectory_visualizer.wandb_initialized:
                    # Create a step number for logging
                    step = len(self.state_data)
                    
                    # Log only input joints and prediction joints (simplified for charts)
                    wandb_data = {
                        # Input joints (renamed from joint_X to input_jointX)
                        "input_joint1": state[0],
                        "input_joint2": state[1],
                        "input_joint3": state[2],
                        "input_joint4": state[3],
                        "input_joint5": state[4],
                        "input_joint6": state[5],
                        "input_gripper": state[6] if len(state) > 6 else 0,
                    }
                    
                    # Add prediction data if available
                    if prediction is not None:
                        pred_data = {
                            "pred_joint1": prediction[0],
                            "pred_joint2": prediction[1],
                            "pred_joint3": prediction[2],
                            "pred_joint4": prediction[3],
                            "pred_joint5": prediction[4],
                            "pred_joint6": prediction[5],
                            "pred_gripper": prediction[6] if len(prediction) > 6 else 0,
                        }
                        wandb_data.update(pred_data)
                    
                    # ---> MODIFIED: Combine images and log <---
                    combined_image_rgb = None
                    if wrist_img_rgb is not None and head_img_rgb is not None:
                        # Ensure same height before concatenating
                        h1, w1, _ = wrist_img_rgb.shape
                        h2, w2, _ = head_img_rgb.shape
                        if h1 != h2:
                            # Resize head image to wrist image height
                            new_w2 = int(w2 * (h1 / h2))
                            head_img_resized = cv2.resize(head_img_rgb, (new_w2, h1), interpolation=cv2.INTER_AREA)
                            combined_image_rgb = cv2.hconcat([wrist_img_rgb, head_img_resized])
                        else:
                             combined_image_rgb = cv2.hconcat([wrist_img_rgb, head_img_rgb])
                    elif wrist_img_rgb is not None:
                         combined_image_rgb = wrist_img_rgb # Only wrist image available
                    
                    # Log combined image if available
                    if combined_image_rgb is not None:
                        wandb_data["camera_views"] = wandb.Image(combined_image_rgb, caption="Wrist (left) & Head (right)")
                    # ---> END MODIFIED <---

                    # Log metrics and combined image (if available)
                    wandb.log(wandb_data)
                    
                    # For every 5 points, update the full 3D visualization and log interactive plot to media
                    if len(self.state_data) % 5 == 0 or len(self.state_data) <= 2:
                        # Generate the figure
                        fig = self.trajectory_visualizer.log_trajectory() # Now returns fig
                        # Log the interactive plot to media if figure was generated
                        if fig is not None:
                            # Only log the interactive visualization, no additional metrics
                            wandb.log({"Trajectory_Interactive": wandb.Plotly(fig)})
                    
                    # 周期性地提醒用户可视化链接
                    if len(self.state_data) % 20 == 0 and self.trajectory_visualizer.wandb_url:
                        print(f"\n📊 数据点已保存: {len(self.state_data)} 条记录")
                        print(f"🔗 在浏览器中查看实时可视化: {self.trajectory_visualizer.wandb_url}\n")
            except Exception as e:
                logger.error(f"Error logging to WandB: {e}")
                logger.error(traceback.format_exc())
        
        logger.info(f"Saved data to folder: {self.current_folder_path}, timestamp: {timestamp}")
        logger.info(f"End position: {end_position}, Predicted end position: {pred_end_position if prediction is not None else 'N/A'}")
        
        return self.current_folder_path 
# Robot Morphology Optimization System

This project uses multi-objective evolutionary algorithms (NSGA-II) to optimize robot morphology design for better movement performance. The system supports complex robot structures, diverse terrain environments, and realistic physical simulation.

## Features

- Using genetic encoding to represent robot morphology:
  - Multiple joint types (rotational, fixed, prismatic, spherical, continuous)
  - Various geometric shapes (box, cylinder, sphere, capsule)
  - Different materials (metal, plastic, rubber)
  - Adjustable dimensions and physical parameters
- Realistic physical simulation based on PyBullet physics engine
- Multiple terrain environments:
  - Flat ground
  - Stairs
  - Random uneven terrain
  - Terrain with obstacles
- Multi-objective optimization, considering:
  - Distance to target point (minimize)
  - Motion path linearity (maximize)
  - Stability/tilt during movement (minimize)
  - Energy efficiency (minimize)
- Generation of Pareto front to show trade-offs between different designs
- Visualization of the best designs in 3D models and movement performance, including real-time trajectory display

## Dependencies

- NumPy
- PyBullet
- Matplotlib
- Pymoo (evolutionary algorithm library)

## Usage

### Running Optimization

```bash
python test33.py
```

This will start the optimization process and generate the following files:
- `pareto_front.png`: 3D visualization of the Pareto front
- `best_distance.urdf`: Best design for minimizing distance to target
- `best_linearity.urdf`: Best design for maximizing path linearity
- `best_stability.urdf`: Best design for minimizing tilt
- `best_energy.urdf`: Best design for minimizing energy consumption

### Visualizing Robot Design

```bash
python visualize_robot.py best_distance.urdf
```

You can specify simulation time and terrain type with parameters:

```bash
python visualize_robot.py best_linearity.urdf --time 10 --terrain stairs
```

Available terrain types:
- `flat`: Flat ground (default)
- `stairs`: Stair terrain
- `rough`: Random uneven terrain
- `obstacles`: Terrain with obstacles

## Working Principle

1. Using genetic encoding to represent various robot design parameters (joint types, geometric shapes, dimensions, materials, etc.)
2. Describing the physical structure of the robot through URDF files
3. Simulating robot movement performance in the PyBullet physics engine
4. Using the NSGA-II algorithm for multi-objective optimization
5. Generating a Pareto front to show the trade-offs between different design objectives

## Future Improvements

- Integrate ROS MoveIt framework to implement more complex motion planning
- Use real object models from the PyBullet URDF Models library as robot components
- Reference UTIAS STARS' UR5 and Robotiq gripper models to implement more realistic mechanical structures
- Implement more complex control strategies and task evaluations
- Add more terrain adaptability tests
- Optimize computational efficiency and support parallel simulation 
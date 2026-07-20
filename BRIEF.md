## Project Brief: Retro Pizza Hut Roof AR Experience

Develop a browser-based augmented reality experience that identifies and reconstructs the iconic retro Pizza Hut roof.

Using the device camera, motion sensors, and SLAM-style tracking, the experience will guide users as they scan an existing or former Pizza Hut building. Because these buildings vary in dimensions, condition, and appearance—and many former locations have repainted roofs—the detection process must rely primarily on the roof’s distinctive architectural form rather than colour alone.

The system will combine:

- Camera-based architectural feature recognition
- Browser gyroscope and accelerometer data
- Real-time device positioning and environmental tracking
- Constraint-based estimation of a simplified roof mesh
- Visual guidance to help the user capture sufficient angles and coverage

The resulting geometry does not need to reproduce every physical detail. It should generate a stable, simplified mesh that fits the observed building while preserving the recognisable proportions and silhouette of the classic Pizza Hut roof.

The experience will be implemented in TypeScript using React and Three.js, without A-Frame.

using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Linq;

public class CustomAgent : Agent {

    public int seed = 50;

    // Robot physics parameters (angular and linear velocity constant)
    public float angularStep;
    public float linearStep;

    // Flags for randomization
    public bool randomizeAgentRotation = true;
    public bool randomizeAgentPosition = true;
    public bool randomizeTarget = true;
    public float targetRandomArea = 1.8f;
    public float distanceNormFact = 3.0f;

    // The target transform
    private Transform target;

    // Basic starting position/rotation for the agent
    private Vector3 startingPos;
    private Quaternion startingRot;

    // Basic starting position/rotation for the target
    private Vector3 startingPosTarget;
    private Quaternion startingRotTarget;

    // Lists for obstacles and chargers
    private GameObject[] obstacleList;
    private GameObject[] chargerList;

    // Reward support variable
    private float oldDistance;

    // Called once at the beginning
    public override void Initialize() {

        Random.InitState(seed);

        // Find target, obstacles, and chargers via tag
        target = GameObject.FindGameObjectWithTag("Target").transform;
        obstacleList = GameObject.FindGameObjectsWithTag("Obstacle");
        chargerList = GameObject.FindGameObjectsWithTag("Charger");

        // Save starting positions and rotations for agent and target
        startingPos = transform.position;
        startingRot = transform.rotation;
        startingPosTarget = target.position;
        startingRotTarget = target.rotation;

        // Compute the initial distance from the target
        oldDistance = Vector3.Distance(target.position, transform.position);
    }

    // Called at the beginning of each episode
    public override void OnEpisodeBegin() {
        Debug.Log("Episode Begin: Resetting Environment");
        randomizeTarget = true;

        //Random.InitState(seed);

        // --- Randomize the target position ---
        if (randomizeTarget) {
            do {
                target.position = new Vector3(
                    Random.Range(0.5f, 9.5f),
                    0f,
                    Random.Range(0.5f, 9.5f)
                );
            } while (VerifyIntersectionWithObstacles(target.gameObject));
        }

        // --- Randomize charger positions ---
        // Randomize each charger so that they do not overlap obstacles or each other.
        // foreach (GameObject charger in chargerList) {
        //     do {
        //         charger.transform.position = new Vector3(
        //             Random.Range(-targetRandomArea, targetRandomArea),
        //             0f,
        //             Random.Range(-targetRandomArea, targetRandomArea)
        //         );
        //     } while (VerifyIntersectionWithObstacles(charger));
        // }

        // --- Reset the agent ---
        // Reset agent's position and rotation to their starting values
        transform.position = startingPos;
        transform.rotation = startingRot;

		randomizeAgentRotation = true;
		randomizeAgentPosition = true;

        // Optionally randomize the agent's rotation
        if (randomizeAgentRotation) {
            transform.Rotate(new Vector3(0f, Random.Range(0, 360), 0f));
        }

        // Optionally randomize the agent's position (checking collisions against obstacles and chargers)
        // if (randomizeAgentPosition) {
        //     do {
        //         transform.position = new Vector3(
        //             Random.Range(0, targetRandomArea),
        //             0.5f,
        //             Random.Range(0, targetRandomArea)
        //         );
        //     } while (VerifyIntersectionWithObstacles(this.gameObject));
        // }

        Debug.Log("Agent randomized position: " + transform.position);

        // Recalculate the initial distance from the target
        oldDistance = Vector3.Distance(target.position, transform.position);
    }

    // Called when actions are received (from the neural network or heuristic)
    public override void OnActionReceived(ActionBuffers actionBuffers) {

        if(actionBuffers.ContinuousActions.Length < 2) 
            return;

        float speed = actionBuffers.ContinuousActions[0];
        float desiredAngle = actionBuffers.ContinuousActions[1];

        // // Apply absolute rotation
        // transform.Rotate(0f, desiredAngle, 0f);
        // // Move forward based on current rotation
        // transform.Translate(Vector3.forward * speed);ù

        float radians = desiredAngle * Mathf.Deg2Rad;

        float deltaX = speed * Mathf.Cos(radians);
        float deltaZ = speed * Mathf.Sin(radians);

        // Update position directly
        transform.position += new Vector3(deltaX, 0f, deltaZ);
        
        // Set absolute rotation
        transform.rotation = Quaternion.Euler(0f, desiredAngle, 0f);

        

    // Debug.Log("Linear: " + moveAction + " , Target Angle: " + desiredAngle);

    // // Move forward
    // float v = 0.0f;
    // float t = 0.0f;
    // if (!float.IsNaN(moveAction)){
    //     v = moveAction;
    // }
    // if(!float.IsNaN(desiredAngle)) {
    //     float currentYAngle = transform.eulerAngles.y;
    //     t = Mathf.DeltaAngle(currentYAngle, desiredAngle * Mathf.Rad2Deg);
    //     // transform.Rotate(Vector3.up * angularVelocity);
    //     // transform.rotation = Quaternion.Euler(0, , 0);
    //     transform.rotation = Quaternion.Euler(0, desiredAngle * Mathf.Rad2Deg, 0);
    // }
    // transform.Translate(Vector3.forward * v);
    // transform.Rotate(Vector3.up * t);

    // Quaternion targetRotation = Quaternion.Euler(0, desiredAngle, 0);
    // float rotationSpeed = 180f;
    // transform.rotation = Quaternion.RotateTowards(transform.rotation, targetRotation, rotationSpeed * Time.deltaTime);
    // float desiredAngleInDegrees = desiredAngle * Mathf.Rad2Deg;

    // Instantly set the rotation to the absolute angle (in degrees)
    
    }

    // Observation helper function: calculates normalized distance and angle.
    private (float, float) CalculateDistanceAndAngle(Vector3 from, Vector3 to, Vector3 forward, Vector3 up, float normFactor) {
        Vector2 fromPos = new Vector2(from.x, from.z);
        Vector2 toPos = new Vector2(to.x, to.z);
        float distance = Vector2.Distance(fromPos, toPos) / normFactor;

        Vector3 direction = to - from;
        float angle = (0.5f - (Vector3.SignedAngle(direction, forward, up) / 360f));
        return (distance, angle);
    }

    public override void CollectObservations(VectorSensor sensor) {
        // Calculate observations to the target
        (float targetDistance, float targetAngle) = CalculateDistanceAndAngle(
            transform.position, target.position, transform.forward, transform.up, distanceNormFact);
        sensor.AddObservation(targetAngle);
        sensor.AddObservation(targetDistance);

        // Find the nearest charger and observe its angle and distance
        GameObject nearestCharger = null;
        float minDistance = float.MaxValue;
        foreach (GameObject charger in chargerList) {
            float dist = Vector3.Distance(transform.position, charger.transform.position);
            if (dist < minDistance) {
                minDistance = dist;
                nearestCharger = charger;
            }
        }

        float chargerDistance = 0f;
        float chargerAngle = 0f;
        if (nearestCharger != null) {
            (chargerDistance, chargerAngle) = CalculateDistanceAndAngle(
                transform.position, nearestCharger.transform.position, transform.forward, transform.up, distanceNormFact);
        }
        sensor.AddObservation(chargerAngle);
        sensor.AddObservation(chargerDistance);

        // STL state requirements
        sensor.AddObservation(transform.position[0]); // Agent x
        sensor.AddObservation(transform.position[2]); // Agent y
        sensor.AddObservation(target.position[0]); // Target x
        sensor.AddObservation(target.position[2]); // Target y
        sensor.AddObservation(nearestCharger.transform.position[0]); // Nearest charger x
        sensor.AddObservation(nearestCharger.transform.position[2]); // Nearest charger y
    }

    // Heuristic method for keyboard control
    public override void Heuristic(in ActionBuffers actionsOut) {
        int action = 0;
        if (Input.GetKey(KeyCode.A)) action = 1;
        if (Input.GetKey(KeyCode.D)) action = 2;

        var actions = actionsOut.DiscreteActions;
        actions[0] = action;
    }

    // When a collision with a solid object occurs
    private void OnCollisionEnter(Collision collision) {
        if (collision.collider.CompareTag("Obstacle") || collision.collider.CompareTag("Wall"))
            SetReward(-1f);
    }

    // When staying within a trigger (for the target)
    private void OnTriggerStay(Collider collision) {
        if (collision.CompareTag("Target"))
            SetReward(1f);
    }

    // This method uses LINQ to combine the obstacles and chargers.
    // It checks if the given GameObject's renderer bounds intersect with any
    // of the other objects’ renderer bounds.
    private bool VerifyIntersectionWithObstacles(GameObject gO) {
        // Combine obstacles and chargers into one array.
        GameObject[] allObjects = obstacleList.Concat(chargerList).ToArray();

        // Get the renderer for the given object.
        Renderer myRenderer = gO.GetComponentInChildren<Renderer>();
        if (myRenderer == null)
            return false; // No renderer to check against.

        foreach (GameObject obj in allObjects) {
            // Skip self–comparison.
            if (obj == gO)
                continue;

            Renderer otherRenderer = obj.GetComponentInChildren<Renderer>();
            if (otherRenderer != null && myRenderer.bounds.Intersects(otherRenderer.bounds))
                return true;
        }
        return false;
    }
}

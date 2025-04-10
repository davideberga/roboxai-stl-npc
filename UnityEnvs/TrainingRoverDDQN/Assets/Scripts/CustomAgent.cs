using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Linq;

public class CustomAgent : Agent {

    public int seed = 40;


    // Robot physics parameters (angular and linear velocity constant)
    public float angularStep;
    public float linearStep;

    // Flags for randomization
    public bool randomizeAgentRotation = false;
    public bool randomizeAgentPosition = false;
    public bool randomizeTarget = false;
    public float targetRandomArea = 1.8f;
    public float distanceNormFact = 6.0f;

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

    // List of initial agent positions
    private Vector3[] initialPositions = new Vector3[5];
    private int currentPositionIndex = 0;  
    private float y_pos = 0.09f; // y position of the agent (fixed)

    // Called once at the beginning
    public override void Initialize() {

        Random.InitState(seed);

        // The 5 possible initial positions for the agent
        initialPositions[0] = new Vector3(0f, y_pos, -1.85f);
        initialPositions[1] = new Vector3(1.94f, y_pos, 1.58f);
        initialPositions[2] = new Vector3(-1.89f, y_pos, 0.6f);
        initialPositions[3] = new Vector3(-0.97f, y_pos, -0.47f);
        initialPositions[4] = new Vector3(1.99f, y_pos, 0.64f);

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
        float targetRandomArea = 1.8f;

        // --- Randomize the target position ---
        if (randomizeTarget) {
            do {
                target.position = new Vector3(
                    Random.Range(-targetRandomArea, targetRandomArea),
                    0f,
                    Random.Range(-targetRandomArea, targetRandomArea)
                );
            } while (verifyIntersectionWithObstacle(target.gameObject));
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
        //     } while (verifyIntersectionWithObstacle(charger));
        // }

        // --- Reset the agent ---
        // Reset agent's position and rotation to their starting values
        //transform.position = startingPos;
        transform.position = initialPositions[currentPositionIndex];
        currentPositionIndex = (currentPositionIndex + 1) % initialPositions.Length;  
        transform.rotation = startingRot;

		randomizeAgentRotation = false;
		randomizeAgentPosition = false;

        // Optionally randomize the agent's rotation
        if (randomizeAgentRotation) {
            transform.Rotate(new Vector3(0f, Random.Range(0, 360), 0f));
        }

        // Optionally randomize the agent's position (checking collisions against obstacles and chargers)
        if (randomizeAgentPosition) {
            do {
                transform.position = new Vector3(
                    Random.Range(-targetRandomArea, targetRandomArea),
                    0f,
                    Random.Range(-targetRandomArea, targetRandomArea)
                );
            } while (verifyIntersectionWithObstacle(this.gameObject));
        }

        Debug.Log("Agent randomized position: " + transform.position);

        // Recalculate the initial distance from the target
        oldDistance = Vector3.Distance(target.position, transform.position);
    }

    // Called when actions are received (from the neural network or heuristic)
    public override void OnActionReceived(ActionBuffers actionBuffers) {
        var actionBuffer = actionBuffers.DiscreteActions;
        float angularVelocity = 0f;
        float linearVelocity = linearStep;

        if (actionBuffer[0] == 1) { // turn right
            angularVelocity = angularStep;
            linearVelocity = 0f;
        }
        if (actionBuffer[0] == 2) { // turn left
            angularVelocity = -angularStep;
            linearVelocity = 0f;
        }

        transform.Rotate(Vector3.up * angularVelocity);
        transform.Translate(Vector3.forward * linearVelocity);
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
        distanceNormFact = 6.0f;
        // Debug.Log(distanceNormFact);
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
    // Utility function to check if there is an intersection between the input object
	// and one of the obstacles
	private bool verifyIntersectionWithObstacle( GameObject gO ) {
		// Iterate over the list of the Obstacle
		foreach( GameObject obstacle in obstacleList )
			if( obstacle.GetComponent<Renderer>().bounds.Intersects( gO.GetComponentInChildren<Renderer>().bounds ) ) 
				return true;
		return false;
	}
}

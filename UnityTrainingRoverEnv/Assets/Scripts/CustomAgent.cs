using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;


// Declaration of the main class for the agent, that inherited from the Agent class of Unity-ML Agents
public class CustomAgent : Agent {

	// Robot physics parameters (angular and linear velocity constant)
	public float angularStep;
	public float linearStep;

	// Variables for the initial position of the target
	public bool randomizeAgentRotation = true;
	public bool randomizeAgentPosition = true;
	public bool randomizeTarget = true;
	public float targetRandomArea = 1.8f;
	public float distanceNormFact = 3.0f;

	// The object that represent the target
	private Transform target;

	// Basic starting position/rotation of the agent for the reset
	// after every episode
	private Vector3 startingPos;
	private Quaternion startingRot;

	// Basic starting position/rotation of the target for the reset
	// after every episode
	private Vector3 startingPosTarget;
	private Quaternion startingRotTarget;

	// List of all the Obstacle to avoid
	private GameObject[] obstacleList;

	// List of all the Charger, to go
	private GameObject[] chargerList;

	// Reward support varaibles
	private float oldDistance;


	// Called at the creation of the enviornment (before the first episode)
	// and only once
	public override void Initialize() {

		// Fill the game object target searching for the tag (setted in the editor)
		target = GameObject.FindGameObjectWithTag("Target").transform;

		// Fill the list of the Obstacle searching for the tag (setted in the editor)
		obstacleList = GameObject.FindGameObjectsWithTag("Obstacle");

		// Fill the list of the Charger searching for the tag (setted in the editor)
		chargerList = GameObject.FindGameObjectsWithTag("Charger");


		// Setting the basic rotation and position
		startingPos = transform.position;
		startingRot = transform.rotation;

		// Setting the basic rotation and position of the target
		startingPosTarget = target.transform.position;
		startingRotTarget = target.transform.rotation;

		// Compute the initial distance from the target
		oldDistance = Vector3.Distance( target.position, transform.position );
	}


	// Called at the every new episode of the training loop,
	// after each reset (both from target, crash or timeout)
	public override void OnEpisodeBegin() {

		// Randomize the position of the target, iterate check to avoid compenetration
		// between the target and the Obstacle
		//target.GetComponentInChildren<MeshRenderer>().enabled = false;
		if( randomizeTarget ) {
			do {
				target.position = new Vector3(Random.Range(-targetRandomArea, targetRandomArea), 0.0f, Random.Range(-targetRandomArea, targetRandomArea));	
			} while ( verifyIntersectionWithObstacle( target.gameObject ) );
		}
		// Reset the position of the agent to the basic settings
		// at the beginning of each episode
		transform.position = startingPos;
		transform.rotation = startingRot;
		// If the flag is active randomize the initial agent rotation at each episode
		if( randomizeAgentRotation ) transform.Rotate( new Vector3(0f, Random.Range(0, 360), 0f) );
		// If the flag is active randomize the initial agent position at each episode
		if( randomizeAgentPosition ) {
			do {
				transform.position = new Vector3(Random.Range(-targetRandomArea, targetRandomArea), 0.0f, Random.Range(-targetRandomArea, targetRandomArea));	
			} while ( verifyIntersectionWithObstacle( this.gameObject ) );
		}
		// Compute the initial distance from the target
		oldDistance = Vector3.Distance( target.position, transform.position );
	}


	// Listener for the action received, both from the neural network and the keyboard
	// (if heuristic mode), inside the Python script, the action is passed with the step funciton
	public override void OnActionReceived(ActionBuffers actionBuffers)	{

		// Read the action buffer, in this set-up, discrete
		var actionBuffer = actionBuffers.DiscreteActions;
		// Basic setting for the action 0 (CoC)
		float angularVelocity = 0f;
		float linearVelocity = linearStep;
		// Listener for action 1, turn right
		// change angular and lienar velocity
		if ( actionBuffer[0] == 1 ) {
			angularVelocity = angularStep;
			linearVelocity = 0f;
		}
		// Listener for action 2, turn left	
		// change angular and lienar velocity
		if ( actionBuffer[0] == 2 ) {
			angularVelocity = -angularStep;
			linearVelocity = 0f;
		}
		// Apply the movement (rotation and translation) according with angular and linear velocity
		//transform.Rotate(Vector3.up * Time.deltaTime * angularVelocity);
		//transform.Translate(Vector3.forward * Time.deltaTime * linearVelocity);
		transform.Rotate(Vector3.up * angularVelocity);
		transform.Translate(Vector3.forward * linearVelocity);	
	}


	// Listener for the observations collections.
	// The observations for the LiDAR sensor are inherited from the 
	// editor, in thi function we add the other observations (angle, distance)
	private (float, float) CalculateDistanceAndAngle(Vector3 from, Vector3 to, Vector3 forward, Vector3 up, float normFactor) {
		Vector2 fromPos = new Vector2(from.x, from.z);
		Vector2 toPos = new Vector2(to.x, to.z);
		float distance = Vector2.Distance(fromPos, toPos) / normFactor;
		
		Vector3 direction = to - from;
		float angle = (0.5f - (Vector3.SignedAngle(direction, forward, up) / 360f));
		
		return (distance, angle);
	}

	public override void CollectObservations(VectorSensor sensor) {
		// Calculate normalized distance and angle to target
		(float targetDistance, float targetAngle) = CalculateDistanceAndAngle(transform.position, target.position, transform.forward, transform.up, distanceNormFact);
		
		// Find nearest charger
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
			(chargerDistance, chargerAngle) = CalculateDistanceAndAngle(transform.position, nearestCharger.transform.position, transform.forward, transform.up, distanceNormFact);
		}
		
		// Add observations
		sensor.AddObservation(targetAngle);
		sensor.AddObservation(targetDistance);
		sensor.AddObservation(chargerAngle);
		sensor.AddObservation(chargerDistance);
	}



	// Debug function, useful to control the agent with the keyboard in heurisitc mode
	// (must be setted in the editor)
	public override void Heuristic(in ActionBuffers actionsOut) { 

		// Set the basic action and wait or a keyboard key
		int action = 0;
		if (Input.GetKey(KeyCode.A)) action = 1;
		if (Input.GetKey(KeyCode.D)) action = 2;
		// Add the action to the actionsOut object
		var actions = actionsOut.DiscreteActions;
		actions[0] = action;
	}

	
	// Listener for the collison with a solid object
	private void OnCollisionEnter(Collision collision) { 

		// Check if the collision is within an obstacle (avoid activation with the floor)
		// or with a wall, the end of the episode is now managed by the wrapper.
		// Set the reward base value for a crash
		if (collision.collider.CompareTag("Obstacle") || collision.collider.CompareTag("Wall")) SetReward(-1f);
	}


	// Listener for the collison with a trigger (non-solid) object, the end of the 
	// episode is now menaged by the wrapper.
	private void OnTriggerStay(Collider collision) { 

		// Check collision with the target and set reward base value for a success
		if (collision.CompareTag("Target")) SetReward(1f);
	}


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
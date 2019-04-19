# DanganAI
Based on the visual novel game Danganronpa by Spike Chunsoft. DanganAI aims to pit AI controlled students in a class trial given that one of the AI students is a murderer. This simulation aims to determine the multitude of ways that a student can survive in a class trial

### Requirements
* ~~Tensorflow~~
* NumPy
* Python


### How it Works
16 unique AI agents or students in this case students will be locked inside a class trial. The agents will have 4 actions. Accuse another agent of the deed, Agree or Disagree with the accusation or state some evidence. These actions are dependent on the previous evidence presented and the previous action of the agent.  With these 2 variables the agent can come up with a hopefully meaningful decision and contribution for the trial. 

After the trial is when the agents are evaluated based on their decisions during the trial. The agents would be divided into 2 categories according to their evaluation scores. Each agent in each category would have a "genetic" crossover with a randomly selected agent. 


### What's New?

* #### Version 0.2
	* Removed the Tensorflow Requirement. All computation is now based purely on NumPy



### To-Do List
* Implement stats per student
* Build the necessary GUI component of the project 
* Optimize the program

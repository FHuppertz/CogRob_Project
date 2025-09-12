Repetitions of the same task with dfifferent models and with or without memory for quantitative results

Trying with variations of the world with different location locations for the same task

Attempt different tasks (in the sense of having the same task but with different phrasing)
    e.g. clean up and put the objects away
As well as different tasks (not all would be putting things away?)
    50-50 different phrasing and different tasks

Do we need to monitor this manually? Automated tests?
    PyBullet location tests

Declare the modifications and changes made to Qwen if we do end up using Qwen for the evaluation

ICRA

-----------------
during tests, check number of tool calls and success (model belief and actual truth) as metrics

potential tasks:
    put stuff in shelf
    swap items
    leave room
    reorder in shelf

world:
    standard world
    randomized world (wish)


Fact that models are trained for single task conversations, where at the end of the task, a new task is made with an empty context. Training so also means that models are sometimes even incapable of switching tasks in a single context. Tool calling is less common, and training environments often have agentic models performing one task at a time. But for an embodied cognitive agent, it becomes important to be able to keep the world in context and perform multiple tasks in it.


Devstral would never pick up the box, and prompting it with "Put away the mug, cube and box into the shelf." does not help, as it gets into a loop of being short sighted and always putting things on the floor to pick up the next item it comes across, and almost never into the shelf. Even on calling the end_task tool, it would never stop the turn and keep on infinitely generating.

Update: Devstral seems to prefer just two items at a time. Sometimes the mug and the cube, or the cube and the box, and so on.


I would like to create a diagram for the architeture of the project, which we want to signify the components of the architecture and how they connect and communicate with each other. First, look through the code to come to an understanding of the code, and write an md file with mermaid diagrams detailing the architecture of the project. We have an LLM cognitive core, with episodic memory using a semantic storage, and working memory for planning and reasoning using the scratchpad. The cognitive element uses toolkits to perform actions in the environment, which is the PyBullet environment/world. Once you are done creating this architecture file with mermaid diagrams, please write a tikz LaTeX file that defines this diagram, that would be suitable for a research paper.


Analyze the project's codebase to understand its structure and components. Then, create a comprehensive architecture documentation file (`architecture.md`) using Mermaid syntax to generate detailed diagrams. These diagrams must visually represent the complete system architecture, including the following core components and their interactions:
*   The LLM Cognitive Core.
*   Episodic Memory, illustrating its implementation via a semantic storage system.
*   Working Memory, detailing its role in planning and reasoning (e.g., utilizing a scratchpad mechanism).
*   The various Toolkits available to the cognitive element.
*   The PyBullet Environment/World.

Clearly depict all data flows, communication channels, and dependencies between these components. After completing the Markdown file, generate a corresponding TikZ LaTeX file (`architecture.tex`) that defines a high-quality, publication-ready version of the primary architectural diagram suitable for inclusion in a research paper.



Motivation: Discuss challenge of flexibly performing complex tasks in an open environment, with failure recovery, and so on. Cognitive architectures try to solve this, cite them here, and while these have shown impressive promise, they have typically been evaluated with clearly defined components for rules, memory, planning, semantics, etc. In recent years, LLMs have shown promise in... and shows how an extensible natural language architecture exploiting the extensive world knowledge present in such models for embodied cognitive robotics.
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
# LLM prompts used (Lab 3)

LLM used: Claude (Anthropic).

## Prompt 1: planner (the handout prompt, plus the domain)
> I want to implement a simple planning agent in Python. Represent a state as a set of logical propositions. Each action should contain a name, positive preconditions, negative preconditions, positive effects, and negative effects. An action is applicable if all of its preconditions are satisfied by the current state. When an action is applied, (1) remove its negative effects from the state, then (2) add its positive effects. Use breadth-first search to find a sequence of actions that achieves a specified goal. The program should detect when no plan exists, print the resulting sequence of actions, and print the states reached after each action. Explain the implementation and identify any assumptions you make.
>
> Domain: locations A, B, C with connections A–B and B–C in both directions. Initial state {At(Robot,A), At(Package,A)}; goal {At(Package,C)}. Actions Move(X,Y), PickUp(Package,L), Drop(Package,L) with the preconditions and effects from the lab sheet.

## Prompt 2: independent validator and tests
> Add a function validate_plan(initial, plan, goal) that re-executes a plan from scratch and reports the first action whose preconditions fail, or whether the goal was not reached. Then write tests: (A) the original problem, (B) PickUp removed, which must print "No plan found", (C) extra robot-only actions, with a check that At(Robot,C) is not treated as At(Package,C).

## Prompt 3: self-verification (Task 5)
> For every action in the plan, identify its preconditions and show that those preconditions are satisfied in the state in which the action is executed.

## Changes and verification done by hand
- I checked the handout's example sequence (PickUp at B) with the validator. It is invalid, and I added this as a test.
- I added Test C2 (Drop removed), which checks more directly that the robot being at C is not the goal.
- I added a Prolog whole-plan verifier (`valid_plan/3`) as an independent check.

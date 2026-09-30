% CSF407 Logic Lab - Tasks 6 and 7: Prolog as an independent plan verifier.

% ---- Task 6: warehouse connectivity ----
connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).

can_move(X,Y) :- connected(X,Y).

% ---- Task 7: checking proposed moves ----
valid_move(X,Y) :- connected(X,Y).

% ---- Extension: verify a WHOLE plan by simulating the state ----
% A state is state(RobotLoc, PackageLoc) where PackageLoc is a location or held.
step(state(X,P), move(X,Y),   state(Y,P))    :- valid_move(X,Y).
step(state(X,X), pickup(X),   state(X,held)).
step(state(X,held), drop(X),  state(X,X)).

valid_plan(S, [], S).
valid_plan(S0, [A|As], S) :- step(S0, A, S1), valid_plan(S1, As, S).

% Is Plan valid from the lab's initial state, and does it achieve At(Package,C)?
achieves_goal(Plan) :- valid_plan(state(a,a), Plan, state(_,c)).

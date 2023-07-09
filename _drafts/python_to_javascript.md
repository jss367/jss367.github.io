
From Python to Javascript

## Scopes

Global Scope: Variables defined outside any function, block, or module (like correctAnswers in your case) have global scope.

Function Scope: Variables declared with var within a function are scoped to that function. They can't be accessed from outside that function, but they are accessible within that function and any of its inner functions.

Block Scope: Variables declared with let and const are block-scoped, which means they exist only within the nearest set of curly braces (i.e., {}).

Module Scope: If you're using modules (e.g., with import and export), each module has its own scope. Only the variables, functions, etc. that are exported can be used outside the module.

Remember, in JavaScript, variables are hoisted to the top of their scope. However, with var this includes initialization, whereas let and const are only declared, not initialized, causing a temporal dead zone where they cannot be accessed until their declaration.

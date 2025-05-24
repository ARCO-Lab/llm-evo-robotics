#!/usr/bin/env python3

def main():
    """A simple function that prints a greeting and performs a basic calculation."""
    print("Hello, World!")
    
    # Perform a simple calculation
    a = 5
    b = 7
    result = a + b
    print(f"The sum of {a} and {b} is {result}")
    
    # Create a simple list and iterate through it
    fruits = ["apple", "banana", "cherry", "orange"]
    print("List of fruits:")
    for i, fruit in enumerate(fruits, 1):
        print(f"{i}. {fruit}")
    
    print("Simple test completed successfully!")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Debug script to understand Agno Agent requirements
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from agno.agent import Agent
    from agno.run.response import RunResponse
    
    print("=== Agent Base Class Info ===")
    print(f"Agent base classes: {Agent.__bases__}")
    print(f"Agent MRO: {Agent.__mro__}")
    
    # Check what methods Agent has
    agent_methods = [method for method in dir(Agent) if not method.startswith('_')]
    print(f"Agent public methods: {agent_methods}")
    
    # Try to inspect the run method
    import inspect
    try:
        sig = inspect.signature(Agent.run)
        print(f"Agent.run signature: {sig}")
        
        # Get the source if possible
        source = inspect.getsource(Agent.run)
        print(f"Agent.run source:\n{source}")
    except Exception as e:
        print(f"Could not inspect Agent.run: {e}")
    
    print("\n=== RunResponse Info ===")
    print(f"RunResponse base classes: {RunResponse.__bases__}")
    print(f"RunResponse attributes: {[attr for attr in dir(RunResponse) if not attr.startswith('_')]}")
    
    # Try creating a RunResponse
    resp = RunResponse(content="test")
    print(f"RunResponse instance: {resp}")
    print(f"RunResponse type: {type(resp)}")
    
    # Check if it has iterator methods
    print(f"Has __iter__: {hasattr(resp, '__iter__')}")
    print(f"Has __next__: {hasattr(resp, '__next__')}")
    
    # Test if we can iterate
    try:
        for item in resp:
            print(f"Iteration item: {item}")
            break
    except Exception as e:
        print(f"Cannot iterate RunResponse: {e}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

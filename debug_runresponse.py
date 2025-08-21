#!/usr/bin/env python3
"""
Test script to debug the RunResponse issue
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from agno.agent import Agent
    from agno.run.response import RunResponse
    
    print("✅ Successfully imported Agent and RunResponse")
    
    # Test RunResponse creation
    response = RunResponse(content="Test message", content_type="text")
    print(f"✅ RunResponse created: {type(response)}")
    print(f"Content: {response.content}")
    print(f"Content type: {response.content_type}")
    
    # Check if it's iterable
    try:
        for item in response:
            print(f"Item: {item}")
    except TypeError as e:
        print(f"❌ RunResponse is not iterable: {e}")
    
    # Check RunResponse attributes
    print(f"RunResponse attributes: {dir(response)}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

#!/usr/bin/env python3
"""
Test script for the Memory class using ChromaDB.
"""

import shutil
from memory import Memory


def main():
    # Create a memory instance
    memory_dir = "./data/test_chroma_db"
    memory = Memory(memory_dir)
    
    print("Adding memories...")
    
    # Add some test memories
    memory_id1 = memory.add_memory(
        "I picked up a red cube from the living room.",
        {"action": "grab", "object": "red cube", "location": "living room"}
    )
    print(f"Added memory with ID: {memory_id1}")
    
    memory_id2 = memory.add_memory(
        "I placed the box on the kitchen shelf.",
        {"action": "place", "object": "box", "location": "kitchen shelf"}
    )
    print(f"Added memory with ID: {memory_id2}")
    
    memory_id3 = memory.add_memory(
        "The robot moved from the front door to the kitchen area.",
        {"action": "move", "from": "front door", "to": "kitchen area"}
    )
    print(f"Added memory with ID: {memory_id3}")
    
    # Search for memories
    print("\nSearching for 'kitchen'...")
    results = memory.search_memories("kitchen", n_results=5)
    for result in results:
        print(f"ID: {result['id']}")
        print(f"Content: {result['content']}")
        print(f"Metadata: {result['metadata']}")
        print("---")
    
    print("\nSearching for 'moved'...")
    results = memory.search_memories("moved", n_results=5)
    for result in results:
        print(f"ID: {result['id']}")
        print(f"Content: {result['content']}")
        print(f"Metadata: {result['metadata']}")
        print("---")
    
    # Retrieve a specific memory
    print(f"\nRetrieving memory by ID: {memory_id1}")
    retrieved_memory = memory.get_memory_by_id(memory_id1)
    if retrieved_memory:
        print(f"Content: {retrieved_memory['content']}")
        print(f"Metadata: {retrieved_memory['metadata']}")
    
    # Get all memories
    print("\nAll memories:")
    all_memories = memory.get_all_memories()
    for mem in all_memories:
        print(f"ID: {mem['id']}")
        print(f"Content: {mem['content']}")
        print(f"Metadata: {mem['metadata']}")
        print("---")
    
    # Delete the memory directory once the script finishes
    try:
        shutil.rmtree(memory_dir)
        print("\nMemory directory deleted successfully.")
    except Exception as e:
        print(f"\nError deleting memory directory: {e}")


if __name__ == "__main__":
    main()
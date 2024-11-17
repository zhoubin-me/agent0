import launchpad as lp
import torch
import torch.multiprocessing as mp
import numpy as np
import time

# Define shared memory arrays
class SharedMemory:
    def __init__(self):
        # Create shared memory arrays using multiprocessing
        self.data = torch.rand(100).cuda()
        self.data.share_memory_()
        self.flag = mp.Value('i', 0)     # Shared integer flag

# Producer node that writes to shared memory
class Producer:
    def __init__(self, shared_mem):
        self.shared_mem = shared_mem
    
    def run(self):
        while True:
            # Generate some data
            data = torch.rand(100)
            
            # Write to shared memory
            with self.shared_mem.flag.get_lock():
                # Wait if consumer is reading
                while self.shared_mem.flag.value == 1:
                    time.sleep(0.001)
                    
                # Write data
                self.shared_mem.data[:] = data
                self.shared_mem.flag.value = 1
            
            time.sleep(0.1)  # Simulate some work

# Consumer node that reads from shared memory 
class Consumer:
    def __init__(self, shared_mem):
        self.shared_mem = shared_mem
        
    def run(self):
        while True:
            # Read from shared memory
            with self.shared_mem.flag.get_lock():
                # Wait for producer to write
                while self.shared_mem.flag.value == 0:
                    time.sleep(0.001)
                    
                # Read data
                x = self.shared_mem.data.sum().item()
                self.shared_mem.flag.value = 0
                
            # Process the data
            print(f"Read data sum: {x:.2f}")
            time.sleep(0.1)  # Simulate some work

def main():
    # Create shared memory
    shared_mem = SharedMemory()
    
    # Create program
    program = lp.Program("shared_memory_example")
    
    # Add nodes
    program.add_node(lp.PyNode(lambda: Producer(shared_mem).run()), label="producer")
    program.add_node(lp.PyNode(lambda: Consumer(shared_mem).run()), label="consumer")
    
    # Launch
    lp.launch(program)

if __name__ == "__main__":
    main()

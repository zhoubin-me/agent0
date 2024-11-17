import launchpad as lp

from typing import Any
import time

# Define a courier that will handle signals between nodes
class SignalCourier:
    def __init__(self):
        self._stop = False
        self._pause = False
    
    def stop(self):
        self._stop = True
        
    def pause(self):
        self._pause = True
        
    def resume(self):
        self._pause = False
        
    def should_stop(self):
        return self._stop
        
    def should_pause(self):
        return self._pause

# Worker node that performs some task
class Worker:
    def __init__(self, courier: SignalCourier):
        self.courier = courier
        
    def run(self):
        while not self.courier.should_stop():
            if self.courier.should_pause():
                time.sleep(0.1)
                continue
                
            # Do some work
            print("Worker: Processing...")
            time.sleep(1)
        print("Worker: Stopped")

# Controller node that manages signals
class Controller:
    def __init__(self, courier: SignalCourier):
        self.courier = courier
        
    def run(self):
        # Let worker run for 3 seconds
        time.sleep(3)
        
        # Pause the worker
        print("Controller: Pausing worker")
        self.courier.pause()
        time.sleep(2)
        
        # Resume the worker
        print("Controller: Resuming worker") 
        self.courier.resume()
        time.sleep(2)
        
        # Stop the worker
        print("Controller: Stopping worker")
        self.courier.stop()

# Create program
program = lp.Program("async_signals")

# Create shared courier
courier = SignalCourier()

# Add nodes
program.add_node(lp.CourierNode(Worker, courier), label="worker")
program.add_node(lp.CourierNode(Controller, courier), label="controller")

lp.launch(program, launch_type="local_mp", terminal="tmux_session")

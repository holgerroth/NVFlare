# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Decorator-based API for NVFlare FOX framework.

This module provides a simplified API for defining server logic using decorators
instead of inheriting from Strategy classes.

Example:
    @flare.main
    def run(num_rounds=3):
        print(flare.sys_info())
        
        for i in range(num_rounds):
            results = flare.clients.train(i, weights)
"""
import functools
import inspect
from typing import Any, Callable, Optional

from .ctx import Context
from .group import all_clients
from .strategy import Strategy


# Thread-local storage for current context
_current_context: Optional[Context] = None


def _set_context(ctx: Context):
    """Set the current context (internal use only)."""
    global _current_context
    _current_context = ctx


def _get_context() -> Context:
    """Get the current context (internal use only)."""
    if _current_context is None:
        raise RuntimeError("No active context. Make sure you're calling this from within a @flare.main decorated function.")
    return _current_context


class ClientsProxy:
    """Proxy object for calling client methods."""
    
    def __getattr__(self, method_name: str):
        """Dynamically create methods that call all clients."""
        def client_method(*args, **kwargs):
            ctx = _get_context()
            
            # Collect results from all clients
            results = []
            
            def collect_result(result, context: Context):
                results.append(result)
                return None
            
            # Call the method on all clients
            all_clients(
                ctx,
                process_resp_cb=collect_result,
            ).__getattr__(method_name)(*args, **kwargs)
            
            return results
        
        return client_method


# Create singleton instance
clients = ClientsProxy()


def sys_info() -> dict:
    """
    Get system information about the federated learning environment.
    
    Returns:
        dict: System information including number of clients, round number, etc.
    """
    ctx = _get_context()
    return {
        "caller": ctx.caller,
        "callee": ctx.callee,
        "call_id": ctx.call_id,
        "props": ctx.props,
    }


class _FunctionStrategy(Strategy):
    """Internal Strategy wrapper for decorator-based functions."""
    
    def __init__(self, func: Callable, func_args: tuple, func_kwargs: dict):
        self.func = func
        self.func_args = func_args
        self.func_kwargs = func_kwargs
        self.name = func.__name__
    
    def execute(self, context: Context):
        """Execute the decorated function with the given context."""
        # Set the global context for this execution
        _set_context(context)
        
        try:
            # Call the user's function
            result = self.func(*self.func_args, **self.func_kwargs)
            return result
        finally:
            # Clean up context
            _set_context(None)


def main(func: Callable) -> Callable:
    """
    Decorator to mark a function as the main server execution logic.
    
    This decorator converts a regular function into a Strategy that can be
    used with the FOX framework. Inside the decorated function, you can use
    flare.sys_info() and flare.clients.* to interact with the federated
    learning environment.
    
    Args:
        func: The function to decorate
        
    Returns:
        A callable that creates a Strategy when called
        
    Example:
        @flare.main
        def run(num_rounds=3):
            print(flare.sys_info())
            
            for i in range(num_rounds):
                results = flare.clients.train(i, weights)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Return a Strategy instance that can be used by the framework
        return _FunctionStrategy(func, args, kwargs)
    
    return wrapper


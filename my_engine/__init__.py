# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
My Engine.

Simple Package for creating chess engines
"""
from my_engine.uci import UCI
from my_engine.engine import EngineBase

if __name__ == '__main__':
    print("MyEngine v0.1")
    u = UCI("MyEngine")
    e = EngineBase("MyEngine", "Virinas-code")

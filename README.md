University project, which aim is to create a system to recognize digits on live video-stream.

At first you should compile WorkingNetwork into shared library

```bash
	gcc -shared -o WorkingNetwork.so -fPIC WorkingNetwork.c
```

`c.py` - with library written in C  

`p.py` - with library written in Python
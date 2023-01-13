import env
import time

g = env.game2048Env(render_mode='human')

while True:
    g.reset()
    time.sleep(2.5)
    g.render()

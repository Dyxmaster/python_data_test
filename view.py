from graphviz import Digraph

# 创建一个有向图
dot = Digraph(comment='Mutex with Semaphore')

# 定义节点
dot.node('P1_noncrit', 'P1: noncrit')
dot.node('P1_crit', 'P1: crit')
dot.node('P1_wait', 'P1: wait')
dot.node('P2_noncrit', 'P2: noncrit')
dot.node('P2_crit', 'P2: crit')
dot.node('P2_wait', 'P2: wait')

# 定义边
dot.edge('P1_noncrit', 'P1_crit', label='y=y+1')
dot.edge('P1_crit', 'P1_wait', label='y>0: y=y-1')
dot.edge('P1_wait', 'P1_noncrit', label='noncrit')
dot.edge('P2_noncrit', 'P2_crit', label='y=y+1')
dot.edge('P2_crit', 'P2_wait', label='y>0: y=y-1')
dot.edge('P2_wait', 'P2_noncrit', label='noncrit')

# 渲染图像
dot.render('mutex_semaphore', view=True)
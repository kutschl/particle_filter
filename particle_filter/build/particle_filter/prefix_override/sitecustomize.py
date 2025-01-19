import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/lukas/Desktop/larace/pf/particle_filter/install/particle_filter'

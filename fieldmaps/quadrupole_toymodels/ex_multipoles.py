from fit_multipoles_fft import FieldMap

m=FieldMap("field_map_quad_edge_30deg_resol_2mm.txt")

m.plot_bn_z(1,0.01)
m.plot_bn_z(2,0.01)
m.plot_bn_z(3,0.01)
m.plot_bn_z(4,0.01)


m.plot_bn_r(3,0.401)

quadfringe = bpmeth.FieldExpansion(b=("0", "0.1*(tanh(s)+1)/2",))
rot_quadfringe = quadfringe.transform(30/180*np.pi)
cut_quadfringe = quadfringe.cut_at_angle(30/180*np.pi)
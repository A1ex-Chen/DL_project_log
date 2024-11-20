def forward(self, xyz):
    B, _, _ = xyz.shape
    if self.normal_channel:
        norm = xyz[:, 3:, :]
        xyz = xyz[:, :3, :]
    else:
        norm = None
    xyz = xyz.permute(0, 2, 1)
    l1_xyz, l1_points = self.sa1(xyz, norm)
    l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
    l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
    x = l3_points.view(B, 1024)
    x = self.drop1(F.relu(self.bn1(self.fc1(x))))
    x = self.drop2(F.relu(self.bn2(self.fc2(x))))
    return x

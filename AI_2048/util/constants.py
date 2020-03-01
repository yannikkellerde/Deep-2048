TOP=0
RIGHT=1
BOTTOM=2
LEFT=3
BACKMAP = {
    TOP:"TOP",
    RIGHT:"RIGHT",
    BOTTOM:"BOTTOM",
    LEFT:"LEFT"
}
LEFT_INDICES=[range(4),range(4,8),range(8,12),range(12,16)]
TOP_INDICES=[range(0,13,4),range(1,14,4),range(2,15,4),range(3,16,4)]
RIGHT_INDICES=[list(reversed(x)) for x in LEFT_INDICES]
BOTTOM_INDICES=[list(reversed(x)) for x in TOP_INDICES]
INDICES = [TOP_INDICES,RIGHT_INDICES,BOTTOM_INDICES,LEFT_INDICES]

ROTATIONMAPS = [[] for _ in range(8)]
for i in range(16):
    ROTATIONMAPS[0].append(i) # Identity
    ROTATIONMAPS[1].append((3-i%4)*4+i//4) # Right rotate
    ROTATIONMAPS[2].append((i%4)*4+i//4) # 0 to 15 diagonal mirror
    ROTATIONMAPS[3].append((i%4)*4+(15-i)//4) # Left rotate
    ROTATIONMAPS[4].append((3-i%4)*4+(15-i)//4) # 3 to 12 diagonal mirror
    ROTATIONMAPS[5].append(15-i) # 180 Rotate
    ROTATIONMAPS[6].append((i%4)+((15-i)//4)*4) # Horizontal mirror
    ROTATIONMAPS[7].append((3-i%4)+(i//4)*4) # Vertical mirror
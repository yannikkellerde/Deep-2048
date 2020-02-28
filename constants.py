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
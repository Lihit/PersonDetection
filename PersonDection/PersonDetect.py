import darknet as dn

# yolo的一些配置
dn.set_gpu(0)
net = dn.load_net(str.encode("../cfg/yolov2.cfg"),
                  str.encode("../weights/yolo.weights"), 0)
meta = dn.load_meta(str.encode("../cfg/coco.data"))


def detectPerson(ImagePath, thresh=.5, hier_thresh=.5, nms=.45):
    PersonDict = {}
    ret = dn.detect(net, meta, str.encode(ImagePath), thresh, hier_thresh, nms)
    for index, person in enumerate(ret):
        (x, y, w, h) = person[2]
        x -= w/2
        if x < 0:
            x = 0
        y -= h/2
        if y < 0:
            y = 0
        PersonDict['person' + str(index + 1)] = [int(x),int(y),int(w),int(h)]
    return PersonDict

def details(t,loss,dis,vloss,vdis):
    print("time cost:"+str(t/60)+" mins")
    print("valid ave_Loss: "+str(vloss)+" | average distance: "+str(vdis))
    print("train ave_Loss: "+str(loss)+" | average distance: "+str(dis))
    print(" ")
    print("---------------------------------------")

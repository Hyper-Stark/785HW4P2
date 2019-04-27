def details(t,loss,dis,vloss,vdis):
    print("time cost:"+str(t/60)+" mins")
    print("valid ave_Loss: "+str(vloss.item())+" | average distance: "+str(vdis.item()))
    print("train ave_Loss: "+str(loss.item())+" | average distance: "+str(dis.item()))
    print(" ")
    print("---------------------------------------")

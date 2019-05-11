function [an] = propagacionAdelante(capas,W,b,p,func)
    an=p;
    for i=1:capas
        if(func(i,1)==1) 
            an=purelin(W{1,i},an,b{1,i});
        elseif(func(i,1)==2)
            an=logsig(W{1,i},an,b{1,i});
        elseif((func(i,1)==3))
            an=tansig(W{1,i},an,b{1,i});
        end
    end
    

end


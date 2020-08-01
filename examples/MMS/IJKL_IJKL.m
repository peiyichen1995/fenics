function r = IJKL_IJKL(a,b)
N = size(a,1);
r = 0;
for i = 1:N
    for j = 1:N
        for k = 1:N
            for l = 1:N
                r = r+a(i,j,k,l)*b(i,j,k,l);
            end
        end
    end
end
end
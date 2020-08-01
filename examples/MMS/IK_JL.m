function r = IK_JL(a,b)
N = size(a,1);
r = zeros(N,N,N,N);
for i = 1:N
  for j = 1:N
    for k = 1:N
      for l = 1:N
        r(i,j,k,l) = a(i,k)*b(j,l);
      end
    end
  end
end
end
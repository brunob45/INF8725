function ImageSignal = ImageSine2( Amplitude, Angle, Period, Phase, ImageSize )
    
    A = Amplitude;
    Theta = Angle;
    Lambda = Period;
    Phi = Phase;
    
    i = sin( Theta );
    j = cos( Theta );
    
    K = cat( 3, i, j );
    K = K ./ sqrt( i^2 + j^2 );
    K = K * ( 2 * pi / Lambda );
    
    [X Y] = meshgrid( 1:ImageSize, 1:ImageSize );
    
    if size( X, 1 ) < size( X, 2 )
        X = X';
    end
    if size( Y, 1 ) < size( Y, 2 )
        Y = Y';
    end
    
    Positions = cat( 3, X, Y );
    K = repmat( K, size( Positions, 1 ), size( Positions, 2 ) );
    
    ImageSignal = A * cos( dot( K, Positions, 3 ) + Phi );
    
    MinVal = min( ImageSignal(:) );
    MaxVal = max( ImageSignal(:) );
    Range = MaxVal - MinVal;
    ImageSignal = (ImageSignal - MinVal) / Range * 255;

end
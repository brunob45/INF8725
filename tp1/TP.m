%% Nettoyage du Workspace
    clear all;
    close all;
    clc;

%% Exercice 1
    %%Numero 1
       t = linspace(-4,4,1000);
       s = sin(pi*t);
       for i = 1:1000
           s(i) = s(i)/(pi*t(i));
       end

       % Afficher le r�sultat
       figure;
       plot(t,s);
       xlabel('temps (s)');
       title('Signal sur le Temps');

        % Reponse : Fonction sinus cardinal

    %%Numero 2
        t = linspace(-2,2,500);
        s1 = sin(t);
        s2 = sin(3*t)/3;
        s3 = sin(5*t)/5;

        % Afficher le r�sultat
        figure;
        plot(t, s1, '--');
        hold on;
        plot(t, s2, '--');
        plot(t, s3, '--');

    %%Numero 3
        s4 = s1+s2+s3;
        plot(t, s4, 'LineWidth', 2);
        hold off;

    %%Numero 4
        s50 = sumSine(50, t);
        figure;
        %plot(t, s50);

        s500 = sumSine(500, t);
        hold on;
        plot(t, s500);
        hold off;
    
%% Exercice 2
    %%Numero 1
        %R�ponse : les fr�quences observ�es sont 82.5, 3 et 40 Hz. Ces
        %r�sultats sont obtenus en divisant la pulsation par 2*pi.
    %%Numero 2
        figure;
        ex2num2(20,1);
        ex2num2(75,2);
        ex2num2(100,3);
        ex2num2(160,4);
        ex2num2(180,5);
        ex2num2(330,6);

    %%Numero 3
        %R�ponse : Plus la fr�quence d'�chantillonnage est �lev�e, plus les
        %hautes fr�quences sont visibles. Quand la fr�quence diminue, plus on
        %perd dans la pr�cision du signal.

    %%Numero 4
        %R�ponse : les �chantillonnages � 180 et 330Hz, car ces fr�quences
        %d'�chantillonnage sont au moins 2x plus �lev�e que la plus haute
        %fr�quence du signal (82.5Hz).
    
%% Exercice 3
    %%Numero 1
        t = linspace(0,1,250);
        Y1 = 7*sin(2*pi*10*t);
        Y2 = 4*sin(2*pi*25*t+(pi/3));
        Y3 = 3*cos(2*pi*50*t);

        figure;
        plot(t, Y1);
        hold on;
        plot(t, Y2);
        plot(t, Y3);
        hold off;

        xlabel('t');
        ylabel('Y');
        title("Exercice 3.1");
        legend("Y1", "Y2", "Y3");
    
    %%Numero 2
        %Y1: f=10Hz
        %Y2: f=25Hz
        %Y3: f=50Hz
    
    %%Numero 3
        Z = Y1 + Y2 + Y3;
        figure;
        plot(t, Z, 'LineWidth', 2);
        xlabel('t');
        ylabel('Y');
        title("Exercice 3.3");
        %R�ponse : PGCD(10,25,50) = 5Hz, ce qui peut �tre aussi observ�
        %dans le graphique
    
    %%Numero 4
        figure;
        subplot(3,1,1);
        plot(t, fft(Y1));
        subplot(3,1,2);
        plot(t, fft(Y2));
        subplot(3,1,3);
        plot(t, fft(Y3));
        %R�ponse : La TFD�des signaux comporte deux "pics" chaque.
        
    %%Numero 5
        figure;
        plot(t, fft(Z));
        %R�ponse : La TFD du signal composite est l'addition des trois TFD
        %pr�c�dentes.









%% Fonctions
function[] = ex2num2(F, i)
    t = linspace(0,1,F);
    Y = 2*sin(165*pi*t) + 13*cos(6*pi*t) - 3*cos(80*pi*t);
    subplot(2,3,i);
    plot(t, Y);
    xlabel('temps (s)');
    title("�chantillonnage � " + F + "Hz");
end

function[r] = sumSine(n, t)
    r = 1/2;
    for i = 1:n
        e = sin(((2*i)+1)*t);
        e = e / ((2*i)+1);
        r = r + 2*e/pi;
    end
end
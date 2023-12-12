function [keyp,tempos,azimutessession,elevationsession, ...
            confidence_az_th,confidence_az_ch,confidence_el_th,confidence_el_ch] = ...
            Read_files_elevation_Odi_solo(direct,patient, interpolation, vector, percentagebox, coord);

disp("Hey 1pp")

%% Get files
save = 0;
lista = dir(strcat(direct,'\',patient, '\'));
nomes = extractfield(lista,'name');
start = 0;
keypointsfilesind = find(contains(nomes,'keypoints') & ~contains(nomes,'keypointschild')& ~contains(nomes,'keypointsothers'));
keypointsfiles = nomes(1,keypointsfilesind);
keypointsfilesindchild = find(contains(nomes,'keypointschild')& ~contains(nomes,'keypointsothers'));
keypointsfileschild = nomes(1,keypointsfilesindchild);
keypointsfilesindothers = find(contains(nomes,'keypointsothers'));
keypointsfilesothers = nomes(1,keypointsfilesindothers);

jointsfilesind = find(contains(nomes,'jointpoints') & ~contains(nomes,'jointpointschild')& ~contains(nomes,'jointpointsothers'));
jointsfiles = nomes(1,jointsfilesind);
jointsfilesindchild = find(contains(nomes,'jointpointschild')& ~contains(nomes,'jointpointsothers'));
jointsfileschild = nomes(1,jointsfilesindchild);
jointsfilesindothers = find(contains(nomes,'jointpointsothers'));
jointsfilesothers = nomes(1,jointsfilesindothers);

anglesfilesind = find(contains(nomes,'angle_gaze'));
anglesfiles = nomes(1,anglesfilesind);

temposind = find(contains(nomes,'tempos'));
temposfiles = nomes(1,temposind);

vidind = find(contains(nomes,'video1'));
videosfiles = nomes(1,vidind);

keyp = {};
tempos = {};
temposalt = {};
azimutesa = {};
azimutesc = {};
elevationa = {};
elevationc = {};
standard_az_th = {};
standard_az_ch = {};
standard_el_th = {};
standard_el_ch = {};
finallost_mat = {};

ficheiroang = strcat(direct,'\',patient,'\',anglesfiles{1})
ficheirotemp = strcat(direct,'\',patient,'\',temposfiles{1})
foldernamesp = strcat(direct,'\',patient,'\')
%% Read Gaze360 angles file
A = textread(ficheiroang,'%f');
Areshaped = reshape(A,10,size(A,1)/10);
Areshaped(1,:)=Areshaped(1,:)+start;
%% Read frame times file
T = textread(ficheirotemp,'%f');
Treshaped = reshape(T,1,size(T,1)/1);

%% Read keypoints child and adult
    
Breshaped = [];
Dreshaped = [];
Creshaped = [];
Ereshaped = [];

Freshaped = [];
Greshaped = [];

for fileskeyp = 1: size(keypointsfiles,2)

    ficheirokp = strcat(direct,'/',patient,'/',keypointsfiles{1,fileskeyp});
    %ficheirokp = fopen(ficheirokp)
    ficheirokpc = strcat(direct,'/',patient,'/',keypointsfileschild{1,fileskeyp});
    ficheirokpo = strcat(direct,'/',patient,'/',keypointsfilesothers{1,fileskeyp});
    
    ficheiroj = strcat(direct,'/',patient,'/',jointsfiles{1,fileskeyp});
    %ficheiroj = fopen(ficheiroj);
    ficheirojc = strcat(direct,'/',patient,'/',jointsfileschild{1,fileskeyp});
    ficheirojo = strcat(direct,'/',patient,'/',jointsfilesothers{1,fileskeyp});

    % ------ Adult keypoints ------

    B = textread(ficheirokp,'%f');
    Breshaped = [Breshaped reshape(B,32*3+1,size(B,1)/(3*32+1))];

    D = textread(ficheiroj,'%f');
    %JOINTS
    Dreshaped = [Dreshaped reshape(D,32*2+1,size(D,1)/(2*32+1))]; %32 JOINTS
    
    % ------ Child keypoints ------
    
    C = textread(ficheirokpc,'%f');
    %Creshaped = [Creshaped reshape(C,25*3+1,size(C,1)/(25*3+1))]; 
    Creshaped = [Creshaped reshape(C,32*3+1,size(C,1)/(32*3+1))]; %32 joints

    E = textread(ficheirojc,'%f');
    %Ereshaped = [Ereshaped reshape(E,25*2+1,size(E,1)/(2*25+1))];
    Ereshaped = [Ereshaped reshape(E,32*2+1,size(E,1)/(2*32+1))]; %32 joints

    % ------ Other keypoints ------
    
    F = textread(ficheirokpo,'%f');
    %Creshaped = [Creshaped reshape(C,25*3+1,size(C,1)/(25*3+1))]; 
    Freshaped = [Freshaped reshape(F,32*3+1,size(F,1)/(32*3+1))]; %32 joints

    G = textread(ficheirojo,'%f');
    %Ereshaped = [Ereshaped reshape(E,25*2+1,size(E,1)/(2*25+1))];
    Greshaped = [Greshaped reshape(G,32*2+1,size(G,1)/(2*32+1))]; %32 joints

end 

%% ------ Cut adult keypoints and relate with each frame ------

% temposadulto = unique(temposadulto);
% n=1;
% k=1;
% keypointsadulto(:,1,1) = Breshaped(2:end,1);
% for i = 1:36700-1
%     if Breshaped(1,i+1) == Breshaped(1,i)
%          k=k+1;
%         keypointsadulto(:,k,n) = Breshaped(2:end,i+1);
%     else
%         k=1;
%         n = n+1;
%         keypointsadulto(:,k,n) = Breshaped(2:end,i+1);
%     end
% end
% 
% n=1;
% k=1;
% jointsadulto(:,1,1) = Dreshaped(2:end,1);
% for i = 1:36700-1
%     if Dreshaped(1,i+1) == Dreshaped(1,i)
%          k=k+1;
%         jointsadulto(:,k,n) = Dreshaped(2:end,i+1);
%     else
%         k=1;
%         n = n+1;
%         jointsadulto(:,k,n) = Dreshaped(2:end,i+1);
%     end
% end
tini = 0;
tfin = 0;

temposadulto = Breshaped(1,:);
%keypointsadulto = reshape(Breshaped(2:end,:),3,25,size(Breshaped,2));
%jointsadulto = reshape(Dreshaped(2:end,:),2,25,size(Dreshaped,2));
keypointsadulto = reshape(Breshaped(2:end,:),3,32,size(Breshaped,2)); %32 joints
jointsadulto = reshape(Dreshaped(2:end,:),2,32,size(Dreshaped,2));

% Cut keypoints for the 3000 frames
inicio = find(temposadulto >= Treshaped(1)+tini);
fim = find(temposadulto <= Treshaped(end)-tfin);
specificfim = min([fim(end) length(jointsadulto)]);

keypadcut = keypointsadulto(:,:,inicio(1):specificfim)./1000;
keypadcut(1,:)= -keypadcut(1,:);
keypadcut(2,:)= -keypadcut(2,:);
temposadulto = temposadulto(inicio(1):specificfim);

% Find closest frame times to the keypoint times
% for i = 1:length(temposadulto)
%     [minValue(i), closestIndex(i)] = min(abs(temposadulto(i) - Treshaped));
% end
% 
% % Change the duplicate frame values
% [uniqueA i j] = unique(closestIndex,'first');
% indexToDupes = find(not(ismember(1:numel(closestIndex),i)));
% if ismember(length(closestIndex), indexToDupes)
%     indexToDupes(end) = [];
% end
% closestIndex(indexToDupes) = round((closestIndex(indexToDupes-1) + closestIndex(indexToDupes+1))/2);
for i = 1:length(temposadulto)
    closestIndex(i) = find(Treshaped == temposadulto(i));
end
% Keypoints times according to the frames
timesadult = NaN(size(Treshaped));
timesadult(closestIndex) = Treshaped(closestIndex);

% Keypoints according to the frames
%keypad = zeros(3,25,size(Treshaped,2));
keypad = zeros(3,32,size(Treshaped,2)); %32 joints
keypad(:,:,closestIndex) = keypadcut(:,:,:);
keypad(:,:,find(keypad(1,28,:) == 0)) = NaN;

% ------ Adult joints ------

jointadcut = jointsadulto(:,:,inicio(1):specificfim);
jointadcut(:,:,end+1:size(keypadcut,3)) = NaN;
%jointad = zeros(2,25,size(Treshaped,2));
jointad = zeros(2,32,size(Treshaped,2));
jointad(:,:,closestIndex) = jointadcut(:,:,:);
jointad(:,:,find(jointad(1,28,:) == 0)) = NaN;

%% ------ Cut child keypoints and relate with each frame ------

temposchild = Creshaped(1,:);
%keypointschild = reshape(Creshaped(2:end,:),3,25,size(Creshaped,2));
keypointschild = reshape(Creshaped(2:end,:),3,32,size(Creshaped,2));
jointschild = reshape(Ereshaped(2:end,:),2,32,size(Ereshaped,2));


% Cut keypoints for the 3000 frames
inicioc = find(temposchild >= Treshaped(1)+tini);
fimc = find(temposchild <= Treshaped(end)-tfin);
specificfimc = min([fimc(end) length(jointschild)]);

keypchcut = keypointschild(:,:,inicioc(1):specificfimc)./1000;
keypchcut(1,:)=-keypchcut(1,:);
keypchcut(2,:)=-keypchcut(2,:);
temposchild = temposchild(inicioc(1):specificfimc);

% Find closest frame times to the keypoint times
% for i = 1:length(temposchild)
%     [minValuec(i), closestIndexc(i)] = min(abs(temposchild(i) - Treshaped));
% end
% 
% % Change the duplicate values
% [uniqueA i j] = unique(closestIndexc,'first');
% indexToDupes = find(not(ismember(1:numel(closestIndexc),i)));
% if ismember(length(closestIndexc), indexToDupes)
%     indexToDupes(end) = [];
% end
% closestIndexc(indexToDupes) = round((closestIndexc(indexToDupes-1) + closestIndexc(indexToDupes+1))/2);
for i = 1:length(temposchild)
    closestIndexc(i) = find(Treshaped == temposchild(i));
end
% Keypoints times according to the frames
timeschild = NaN(size(Treshaped));
timeschild(closestIndexc) = Treshaped(closestIndexc);

% Keypoints according to the frames
%keypch = zeros(3,25,size(Treshaped,2));
keypch = zeros(3,32,size(Treshaped,2));
keypch(:,:,closestIndexc) = keypchcut(:,:,:);
keypch(:,:,find(keypch(1,28,:) == 0)) = NaN;

% ------ Child joints ------

%jointschild = reshape(Ereshaped(2:end,:),2,25,size(Ereshaped,2));
jointchcut = jointschild(:,:,inicioc(1):specificfimc);
jointch = zeros(2,32,size(Treshaped,2));
jointch(:,:,closestIndexc) = jointchcut(:,:,:);
jointch(:,:,find(jointch(1,28,:) == 0)) = NaN;
% k=strcat(direct,'\',patient,'\')
% drawingskeletons('initial',strcat(direct,'\',patient,'\'),Areshaped,jointad,jointch)
%% ------ Cut other keypoints and relate with each frame ------

temposothers = Freshaped(1,:);
%keypointschild = reshape(Creshaped(2:end,:),3,25,size(Creshaped,2));
keypointsothers = reshape(Freshaped(2:end,:),3,32,size(Freshaped,2));


% Cut keypoints for the 3000 frames
inicioo = find(temposothers >= Treshaped(1)+tini);
fimo = find(temposothers <= Treshaped(end)-tfin);

keypotcut = keypointsothers(:,:,inicioo(1):fimo(end))./1000;
keypotcut(1,:)=-keypotcut(1,:);
keypotcut(2,:)=-keypotcut(2,:);
temposothers = temposothers(inicioo(1):fimo(end));

% Find closest frame times to the keypoint times
% for i = 1:length(temposchild)
%     [minValuec(i), closestIndexc(i)] = min(abs(temposchild(i) - Treshaped));
% end
% 
% % Change the duplicate values
% [uniqueA i j] = unique(closestIndexc,'first');
% indexToDupes = find(not(ismember(1:numel(closestIndexc),i)));
% if ismember(length(closestIndexc), indexToDupes)
%     indexToDupes(end) = [];
% end
% closestIndexc(indexToDupes) = round((closestIndexc(indexToDupes-1) + closestIndexc(indexToDupes+1))/2);
for i = 1:length(temposothers)
    closestIndexo(i) = find(Treshaped == temposothers(i));
end
% Keypoints times according to the frames
timesothers = NaN(size(Treshaped));
timesothers(closestIndexo) = Treshaped(closestIndexo);

% Keypoints according to the frames
%keypch = zeros(3,25,size(Treshaped,2));
keypo = zeros(3,32,size(Treshaped,2));
keypo(:,:,closestIndexo) = keypotcut(:,:,:);
keypo(:,:,find(keypo(1,28,:) == 0)) = NaN;

% ------ Child joints ------

%jointschild = reshape(Ereshaped(2:end,:),2,25,size(Ereshaped,2));
jointsothers = reshape(Greshaped(2:end,:),2,32,size(Greshaped,2));
jointocut = jointsothers(:,:,inicioo(1):fimo(end));
jointo = zeros(2,32,size(Treshaped,2));
jointo(:,:,closestIndexo) = jointocut(:,:,:);
jointo(:,:,find(jointo(1,28,:) == 0)) = NaN;
% k=strcat(direct,'\',patient,'\')
% draw_skeleton(jointad, jointch, jointo, foldernamesp, videosfiles{1})
% drawingskeletons_3d(keypad, keypch, keypo, foldernamesp)



%% 
sum(~isnan(keypad(1,28,:)))
lostchinitial = sum(isnan(keypch(1,28,:)))/size(keypch,3)*100
lostadinitial = sum(isnan(keypad(1,28,:)))/size(keypad,3)*100

%% ------ Plot keypoints and jointpoints ------
%in this setup the mum (other) is placed between the other two subject. So
%I rearrange the data to have proper names
%keypoints

DG =1; 
porta = 0;
if porta ==1
    befo = keypo;
    keypo = keypch;
    keypch = befo;
    
    befco = jointo;
    jointo = jointch;
    jointch = befco;
    
elseif DG == 1

    befch = keypch;
    keypch = keypad;
    keypad = keypo;
    keypo = befch;
    
    befchj = jointch;
    jointch = jointad;
    jointad = jointo;
    jointo = befchj;
end

figure
X = plot(-squeeze(keypch(1,28,:)),squeeze(keypch(3,28,:)),'ob');
hold on
Y = plot(-squeeze(keypad(1,28,:)),squeeze(keypad(3,28,:)),'*r');
Z = plot(-squeeze(keypo(1,28,:)),squeeze(keypo(3,28,:)),'vg');
% A = plot(-squeeze(keypoini(1)),squeeze(keypoini(3)),'vc');
% B = plot(-squeeze(keypchini(1)),squeeze(keypchini(3)),'oc');
% C = plot(-squeeze(keypadini(1)),squeeze(keypadini(3)),'*c');

[x, z] = ginput(3);
keypadini = [-x(1); 0; z(1)];
keypchini = [-x(2); 0 ; z(2)];
keypoini = [-x(3); 0; z(3)];

% for i=1:size(coord,2)
%     Z(i) = plot(coord{i}(1),coord{i}(3),'o');
%     x = [x,i];
% end
xlabel('x')
ylabel('y')
% xlim([-1.5 1])
legend([X Y Z],{'Head child','head adult','head other'})
title('keypoints and target position prima')

% %% Test for correctness of joints reallocation
% 
% 
% figure
% X = plot(-squeeze(jointad(1,28,:)),squeeze(jointad(2,28,:)),'ob');
% hold on
% Y = plot(-squeeze(jointch(1,28,:)),squeeze(jointch(2,28,:)),'*r');
% 
% 
% xlabel('x')
% ylabel('y')
% % xlim([-1.5 1])
% legend([X Y Z],{'Head child','head adult','head other'})
% title('jointpoints')

%% ---- Keep only frames with both skeletons ----
% 
for i=1:size(keypo,3)
   I = peopleid(keypad,jointad, keypadini,keypchini,keypoini,i);
        if I ==2
            keyp2(:,:,i) = keypad(:,:,i);
            joint2(:,:,i) = jointad(:,:,i);
        elseif I ==3 
            keyp3(:,:,i) = keypad(:,:,i);
            joint3(:,:,i) = jointad(:,:,i);
        else
            keyp1(:,:,i) = keypad(:,:,i);
            joint1(:,:,i) = jointad(:,:,i);
        end
   I = peopleid(keypch,jointch, keypadini,keypchini,keypoini,i);
        if I ==1
            keyp1(:,:,i) = keypch(:,:,i);
            joint1(:,:,i) = jointch(:,:,i);
        elseif I ==3 
            keyp3(:,:,i) = keypch(:,:,i);
            joint3(:,:,i) = jointch(:,:,i);
        else
            keyp2(:,:,i) = keypch(:,:,i);
            joint2(:,:,i) = jointch(:,:,i);
        end
  I = peopleid(keypo,jointo, keypadini,keypchini,keypoini,i);
        if I ==1
            keyp1(:,:,i) = keypo(:,:,i);
            joint1(:,:,i) = jointo(:,:,i);
        elseif I == 2
            keyp2(:,:,i) = keypo(:,:,i);
            joint2(:,:,i) = jointo(:,:,i);
        else
            keyp3(:,:,i) = keypo(:,:,i);
            joint3(:,:,i) = jointo(:,:,i);
        end

end

keypad = keyp1;
keypch = keyp2;
keypo= keyp3;
jointad = joint1;
jointch = joint2;
jointo = joint3;

keypad(:,:,find(keypad(1,28,:) == 0)) = NaN;
jointad(:,:,find(jointad(1,28,:) == 0)) = NaN;
keypch(:,:,find(keypch(1,28,:) == 0)) = NaN;
jointch(:,:,find(jointch(1,28,:) == 0)) = NaN;
keypo(:,:,find(keypo(1,28,:) == 0)) = NaN;
jointo(:,:,find(jointo(1,28,:) == 0)) = NaN;




ml = max([size(keypad,3),size(keypch,3), size(keypo, 3)]);
minl = min([size(keypad,3),size(keypch,3), size(keypo, 3)]);

for i =minl:ml
    if size(keypad,3)<ml
       keypad(:,:,i) = NaN(3,32,1);
       jointad(:,:,i) = NaN(2,32,1);
    end

    if size(keypch,3)<ml
       keypch(:,:,i) = NaN(3,32,1);
       jointch(:,:,i) = NaN(2,32,1);
    end

    if size(keypo,3)<ml
       keypo(:,:,i) = NaN(3,32,1);
       jointo(:,:,i) = NaN(2,32,1);
    end
end


% sum(~isnan(keypad(1,28,:)))
% lostchafter = sum(isnan(keypch(1,28,:)))/size(keypch,3)*100
%% da questo video si vede che le persone sono traccate giuste. 
% draw_skeleton(jointad, jointch, jointo, foldernamesp, videosfiles{1})
jointadvideo = sum(~isnan(jointad(1,28,:)))
jointchvideo = sum(~isnan(jointch(1,28,:)))

%% ------ Plot keypoints and jointpoints ------
x = [];
figure
X = plot(-squeeze(keypch(1,28,:)),squeeze(keypch(3,28,:)),'ob');
hold on
Y = plot(-squeeze(keypad(1,28,:)),squeeze(keypad(3,28,:)),'*r');
Z = plot(-squeeze(keypo(1,28,:)),squeeze(keypo(3,28,:)),'vg');
A = plot(-keypoini(1),keypoini(3),'vc');
B = plot(-keypchini(1),keypchini(3),'oc');
C = plot(-keypadini(1),keypadini(3),'*c');
adultrearranged = sum(~isnan(keypad(1,28,:)))
childrearranged = sum(~isnan(keypch(1,28,:)))
adultjointrearranged = sum(~isnan(jointad(1,28,:)))
childjointrearranged = sum(~isnan(jointch(1,28,:)))

% for i=1:size(coord,2)
%     Z(i) = plot(coord{i}(1),coord{i}(3),'o');
%     x = [x,i];
% end
legend([X Y Z],{'Head child','head adult','head other'})
xlabel('x')
ylabel('y')
% xlim([-1.5 1])
title('keypoints and target position dopo')



%% ------ Filtering ------

% Filter keypoints adult
keypad(1,28,:) = medfilt1(keypad(1,28,:), 7);
keypad(2,28,:) = medfilt1(keypad(2,28,:), 7);
keypad(3,28,:) = medfilt1(keypad(3,28,:), 7);
keypad(1,1,:) = medfilt1(keypad(1,1,:), 7);
keypad(2,1,:) = medfilt1(keypad(2,1,:), 7);
keypad(3,1,:) = medfilt1(keypad(3,1,:), 7);
jointad(1,28,:) = medfilt1(jointad(1,28,:), 7);
jointad(2,28,:) = medfilt1(jointad(2,28,:), 7);
jointad(1,1,:) = medfilt1(jointad(1,1,:), 7);
jointad(2,1,:) = medfilt1(jointad(2,1,:), 7);

% Filter keypoints child
keypch(1,28,:) = medfilt1(keypch(1,28,:), 7);
keypch(2,28,:) = medfilt1(keypch(2,28,:), 7);
keypch(3,28,:) = medfilt1(keypch(3,28,:), 7);
keypch(1,1,:) = medfilt1(keypch(1,1,:), 7);
keypch(2,1,:) = medfilt1(keypch(2,1,:), 7);
keypch(3,1,:) = medfilt1(keypch(3,1,:), 7);
jointch(1,28,:) = medfilt1(jointch(1,28,:), 7);
jointch(2,28,:) = medfilt1(jointch(2,28,:), 7);
jointch(1,1,:) = medfilt1(jointch(1,1,:), 7);
jointch(2,1,:) = medfilt1(jointch(2,1,:), 7);

lostchkinectfilter = sum(isnan(keypch(1,28,:)))/size(keypch,3)*100
lostadkinectfilter = sum(isnan(keypad(1,28,:)))/size(keypad,3)*100

adultfiltered = sum(~isnan(keypad(1,28,:)))
childfiltered = sum(~isnan(keypch(1,28,:)))


%% ---- Keypoints interpolation ----

keypad_final = keypad;
keypch_final = keypch;

jointad_final = jointad;
jointch_final = jointch;

if interpolation
    
    for axis1 = 1:size(keypad,1)

        nandataad = squeeze(keypad(axis1,:,:))';
        xdataad = (1:size(nandataad,1))';
        dataad = bsxfun(@(x,y) interp1(y(~isnan(x)),x(~isnan(x)),y), nandataad, xdataad);

        keypad_final(axis1,:,:) = dataad(:,:)';

        nandatach = squeeze(keypch(axis1,:,:))';
        xdatach = (1:size(nandatach,1))';
        datach = bsxfun(@(x,y) interp1(y(~isnan(x)),x(~isnan(x)),y), nandatach, xdatach);

        keypch_final(axis1,:,:) = datach(:,:)';

    end

    for axis1 = 1:size(jointad,1)

        nandataad = squeeze(jointad(axis1,:,:))';
        xdataad = (1:size(nandataad,1))';
        dataad = bsxfun(@(x,y) interp1(y(~isnan(x)),x(~isnan(x)),y), nandataad, xdataad);

        jointad_final(axis1,:,:) = dataad(:,:)';

        nandatach = squeeze(jointch(axis1,:,:))';
        xdatach = (1:size(nandatach,1))';
        datach = bsxfun(@(x,y) interp1(y(~isnan(x)),x(~isnan(x)),y), nandatach, xdatach);

        jointch_final(axis1,:,:) = datach(:,:)';

    end
end

% draw_skeleton(jointad_final, jointch_final, jointo, foldernamesp, videosfiles{1})

%% ------ Plot after interpolation and filtration ------
figure
plot(squeeze(keypch_final(1,28,:)),squeeze(keypch_final(3,28,:)),'ob');
hold on
plot(squeeze(keypad_final(1,28,:)),squeeze(keypad_final(3,28,:)),'*r');
title('Keyp after filtration and interpolation')
legend('head child before','head adult before', 'head child after', 'head adult after')
sum(~isnan(keypad_final(1,28,:)))
lostchinterp = sum(isnan(keypch_final(1,28,:)))/size(keypch_final,3)*100
lostadinterp = sum(isnan(keypad_final(1,28,:)))/size(keypad_final,3)*100

adultinterp = sum(~isnan(keypad_final(1,28,:)))
childinterp = sum(~isnan(keypch_final(1,28,:)))
%% 
if save == 1
    savedir = strcat(direct, '/', patient, '/');
    fileAD = fopen(strcat(savedir,'JOINTSinterp.txt'),'w');
    for i = 1: length(timesadult)
        if ~isnan(jointad_final(1,28,i)) && ~isnan(timesadult(i))
            fprintf(fileAD, '%.7f ', timesadult(i));
            fprintf(fileAD, '%f ', jointad_final(:,:,i));
            fprintf(fileAD, '\n');
        end
    end
    fclose(fileAD);
    fileCH = fopen(strcat(savedir,'JOINTSCHinterp.txt'),'w');
    for i = 1: length(timeschild)
        if ~isnan(jointch_final(1,28,i)) && ~isnan(timeschild(i))
            fprintf(fileCH, '%.7f ', timeschild(i));
            fprintf(fileCH, '%f ', jointch_final(:,:,i));
            fprintf(fileCH, '\n');
        end
    end
    fclose(fileCH);
end
%% Relate Bboxs with Skeletons
    
idxa = [];
idxc = [];
idxo = [];
framea = [];
framec = [];
frameo = [];

% Select data from each subject
a = find(Treshaped>= temposadulto(1,1));
c = find(Treshaped>= temposchild(1,1));
s = max(a(1),c(1));
in = s(1)-1;

for kn = 1:size(jointad_final,3)
    n_bbox = find(Areshaped(1,:) == kn);
   
%     if size(n_bbox,2) >= 2
        
        box_x = {};
        box_y = {};
        box_xx = {};
        box_yy = {};
        notfounda = 1;
        notfoundch = 1;

        for i = 1:length(n_bbox)

             size_x = (Areshaped(9,n_bbox(i))*1920/960 - Areshaped(7,n_bbox(i))*1920/960)*(1+percentagebox);
             size_y = (Areshaped(10,n_bbox(i))*1080/720 - Areshaped(8,n_bbox(i))*1080/720)*(1+percentagebox);
%            size_x = (Areshaped(9,n_bbox(i))*1920/640 - Areshaped(7,n_bbox(i))*1920/640)*(1+percentagebox);
%            size_y = (Areshaped(10,n_bbox(i))*1080/576 - Areshaped(8,n_bbox(i))*1080/576)*(1+percentagebox);

             mean_x = (Areshaped(9,n_bbox(i))*1920/960 + Areshaped(7,n_bbox(i))*1920/960)/2;
             mean_y = (Areshaped(10,n_bbox(i))*1080/720 + Areshaped(8,n_bbox(i))*1080/720)/2;
%             mean_x = (Areshaped(9,n_bbox(i))*1920/640 + Areshaped(7,n_bbox(i))*1920/640)/2;
%             mean_y = (Areshaped(10,n_bbox(i))*1080/576 + Areshaped(8,n_bbox(i))*1080/576)/2;

            posx_left = mean_x - size_x/2;
            posy_up = mean_y - size_y/2;
            posx_right = mean_x + size_x/2;
            posy_down = mean_y + size_y/2;
            
             box_x{end+1} = [Areshaped(7,n_bbox(i))*1920/960, Areshaped(7,n_bbox(i))*1920/960, Areshaped(9,n_bbox(i))*1920/960, Areshaped(9,n_bbox(i))*1920/960, Areshaped(7,n_bbox(i))*1920/960];
             box_y{end+1} = [Areshaped(8,n_bbox(i))*1080/720, Areshaped(10,n_bbox(i))*1080/720, Areshaped(10,n_bbox(i))*1080/720, Areshaped(8,n_bbox(i))*1080/720, Areshaped(8,n_bbox(i))*1080/720];
%             box_x{end+1} = [Areshaped(7,n_bbox(i))*1920/640, Areshaped(7,n_bbox(i))*1920/640, Areshaped(9,n_bbox(i))*1920/640, Areshaped(9,n_bbox(i))*1920/640, Areshaped(7,n_bbox(i))*1920/640];
%             box_y{end+1} = [Areshaped(8,n_bbox(i))*1080/576, Areshaped(10,n_bbox(i))*1080/576, Areshaped(10,n_bbox(i))*1080/576, Areshaped(8,n_bbox(i))*1080/576, Areshaped(8,n_bbox(i))*1080/576];
%             
            box_xx{end+1} = [posx_left, posx_left, posx_right, posx_right, posx_left];
            box_yy{end+1} = [posy_up, posy_down, posy_down, posy_up, posy_up];
            
            % If Therapist
            if jointad_final(1,28,kn) >= posx_left && jointad_final(1,28,kn) <= posx_right ...
                    && jointad_final(2,28,kn) >= posy_up && jointad_final(2,28,kn) <= posy_down && notfounda == 1
                idxa = [idxa n_bbox(i)];
                framea = [framea Areshaped(1,n_bbox(i))];
                notfounda = 0;
            % If Child
            elseif jointch_final(1,28,kn) >= posx_left && jointch_final(1,28,kn) <= posx_right ...
                    && jointch_final(2,28,kn) >= posy_up && jointch_final(2,28,kn) <= posy_down && notfoundch == 1 
                idxc = [idxc n_bbox(i)];
                framec = [framec Areshaped(1,n_bbox(i))];
                notfoundch = 0;
            end
%         end

% plot heads and bounding boxes
%         if ~isnan(jointch_final(1,28,kn)) && ~isnan(jointad_final(1,28,kn)) && mod(kn,100) == 0
%             figure;
%             plot(squeeze(jointch_final(1,28,kn)), squeeze(jointch_final(2,28,kn)), '*', ...
%                 'MarkerSize', 12, 'LineWidth', 2, 'Color', '[0.3010 0.7450 0.9330]');
%             hold on;
%             plot(squeeze(jointad_final(1,28,kn)), squeeze(jointad_final(2,28,kn)), '*', ...
%                 'MarkerSize', 12, 'LineWidth', 2, 'Color', '[0.4660 0.6740 0.1880]');
%             plot(squeeze(jointo(1,28,kn)), squeeze(jointad_final(2,28,kn)), '*', ...
%                 'MarkerSize', 12, 'LineWidth', 2, 'Color', '[1 0 0]');
%             for idx = 1:length(box_x)
%                 plot(box_x{idx}, box_y{idx}, '--', 'LineWidth', 2)
%                 plot(box_xx{idx}, box_yy{idx}, 'LineWidth', 2)
%             end
%             legend('child', 'therapist', 'other', 'BB child', 'BB child2', 'BBadult', 'BBadult2')
% 
% 
%         end

%         % Plot BBoxes and Kinect Heads
%         
%          if in == 500
%               
%             figure;
%             plot(squeeze(jointch_final(1,28,in)), squeeze(jointch_final(2,28,in)), '*', ...
%                 'MarkerSize', 12, 'LineWidth', 2, 'Color', '[0.3010 0.7450 0.9330]');
%             plot(squeeze(jointad_final(1,28,in)), squeeze(jointad_final(2,28,in)), '*', ...
%                 'MarkerSize', 12, 'LineWidth', 2, 'Color', '[0.4660 0.6740 0.1880]');
%             colors = {'[0 0.4470 0.7410]', '[0 0.5 0]', 'k', 'k'};
%              if in == 500     
%                 colors = {'k', '[0 0.4470 0.7410]', '[0 0.5 0]', 'k'};
%             end
%             for idx = 1:length(box_x)
%                 plot(box_x{idx}, box_y{idx}, '--', 'Color', colors{idx}, 'LineWidth', 2)
%                 plot(box_xx{idx}, box_yy{idx}, 'Color', colors{idx}, 'LineWidth', 2)
%             end
%             set ( gca, 'ydir', 'reverse' )
%             xlabel('Horizontal Pixels [px]')
%             ylabel('Vertical Pixels [px]')
%             xlim([0 1920])
%             ylim([0 1080])
% 
%             if in == 500
%                 lgd = legend('Child keypoint', 'Therapist keypoint', 'Other bounding boxes', ...
%                     'Other increased bounding boxes', 'Child bounding box', ...
%                     'Increased Child bounding box', 'Therapist bounding box', ...
%                     'Increased Therapist bounding box', 'Orientation', 'horizontal', 'Location', 'southoutside');
%             else
%                 lgd = legend('Child keypoint', 'Therapist keypoint', 'Child bounding box', ...
%                     'Increased Child bounding box', 'Therapist bounding box', ...
%                     'Increased Therapist bounding box', 'Other bounding boxes', ...
%                     'Other increased bounding boxes', 'Orientation', 'horizontal', 'Location', 'southoutside');
%             end
%             lgd.NumColumns = 2;
%             lgd.FontSize = 16;
%             set(gca,'FontSize',18) % Creates an axes and sets its FontSize to 18
%             title({'Densepose Bounding Boxes and Kinect Head Joints'});
%           end
%     
            end
    in = in+1;
end

matrixadult = NaN(size(Areshaped,1), size(jointad_final,3));
matrixchild = NaN(size(Areshaped,1), size(jointch_final,3));

% matrixadult(:,framea+s(1)) = Areshaped(:,idxa);
% matrixchild(:,framec+s(1)) = Areshaped(:,idxc);
matrixadult(:,framea) = Areshaped(:,idxa);
matrixchild(:,framec) = Areshaped(:,idxc);

%% ------ Filtro, Interpolation ------

% --- Keep only angles and times from frames from densepose with both ---

% matrixadult(:,find(matrixadult(1,:) == 0)) = NaN;
% matrixchild(:,find(matrixchild(1,:) == 0)) = NaN;

matrixadnan = find(isnan(matrixadult(1,:)));
matrixchnan = find(isnan(matrixchild(1,:)));

% matrixadult(:,matrixchnan) = NaN;
% matrixchild(:,matrixadnan) = NaN;

keypad_int = keypad_final;
keypch_int = keypch_final;

keypad_final(:,:,matrixadnan) = NaN(size(keypad_final(:,:,matrixadnan)),'like',keypad_final(:,:,matrixadnan));
keypch_final(:,:,matrixchnan) = NaN(size(keypch_final(:,:,matrixchnan)),'like',keypch_final(:,:,matrixchnan));

jointad_final(:,:,matrixadnan) = NaN(size(jointad_final(:,:,matrixadnan)),'like',jointad_final(:,:,matrixadnan));
jointch_final(:,:,matrixchnan) = NaN(size(jointch_final(:,:,matrixchnan)),'like',jointch_final(:,:,matrixchnan));

lostchgaze360 = sum(isnan(keypch_final(1,28,:)))/size(keypch_final,3)*100
lostadgaze360 = sum(isnan(keypad_final(1,28,:)))/size(keypad_final,3)*100

Afinal = {matrixadult, matrixchild};
start = find(Treshaped>= temposchild(1,1));

finalnumberad= sum(~isnan(keypad_final(1,28,:)))
finalnumberch= sum(~isnan(keypch_final(1,28,:)))

%% print relevant information for analysis
adultrearranged = adultrearranged
adultinterp = adultinterp
childrearranged = childrearranged
childinterp = childinterp
BBa = length(idxa)
BBc = length(idxc)
finalnumberad= sum(~isnan(keypad_final(1,28,:)))
finalnumberch= sum(~isnan(keypch_final(1,28,:)))

%% ------ Filtering gaze angle ------

% Moving average
x = [matrixchild(5,:);matrixchild(6,:);matrixadult(5,:);matrixadult(6,:)];
windowSize = 7; 
b = (1/windowSize)*ones(1,windowSize);
a = 1;
angles=[];
for mx=1:size(x,1)
    angles = [angles;filter(b,a,x(mx,:))];
end

figure;
plot(timesadult,angles(1,:))
hold on
plot(timesadult,angles(2,:))
plot(timesadult,angles(3,:))
plot(timesadult,angles(4,:))
legend('gazech[0]','gazech[1]','gazead[0]','gazead[1]')
title('Gaze from gaze yolo')

                
figure;
X = plot(squeeze(matrixchild(7,:)),squeeze(matrixchild(8,:)),'ob');
hold on
Y = plot(squeeze(matrixadult(7,:)),squeeze(matrixadult(8,:)),'*r');
title('jointpoints BB')
%%
temposf = Treshaped;
keyp{end + 1} = {keypad_final, keypch_final};
tempos{end+1} = temposf;
temposalt{end+1} = temposadulto;

%% Get azimuth angles 

% Get azimuth angles for adult and child
azimutesa{end+1} = angles(3,:);
azimutesc{end+1} = angles(1,:);
elevationa{end+1} = angles(4,:);
elevationc{end+1} = angles(2,:);

confidence_az_th = matrixadult(3,:);
confidence_el_th = matrixadult(4,:);
confidence_az_ch = matrixchild(3,:);
confidence_el_ch = matrixchild(4,:);
%% Join all the gazes of one session
azimutessession = {azimutesa, azimutesc};
elevationsession = {elevationa, elevationc};
% % 
% %     
gaze_file = fopen(strcat(foldernamesp, '\selected_gaze_ch.txt'),'w');
label_file = fopen(strcat(foldernamesp, '\selected_labels.txt'),'w');
for i = 1:size(azimutesa{1},2)
    if ~isnan(keypch_final(1,28,i))
   fprintf(gaze_file,'%f %f %f %f %f %f %d %d %d %d\n',matrixchild(5,i),matrixchild(6,i),confidence_az_ch(i),squeeze(keypch_final(1,28,i)),squeeze(keypch_final(2,28,i)),squeeze(keypch_final(3,28,i)),...
        matrixchild(7,i),matrixchild(8,i),matrixchild(9,i),matrixchild(10,i));
   fprintf(label_file,'%d\n',matrixchild(1,i)-1); 

    end
end
fclose(gaze_file)
fclose(label_file)
%%
% gaze_file = fopen(strcat(foldername, '\selected_gaze_ad.txt'),'w');
% 
% for i = 1:size(azimutesa{1},2) 
%    fprintf(gaze_file,'%f %f %f %f %f %f %d %d %d %d\n',azimutesc{1}(i),elevationc{1}(i),confidence_az_ch(i),squeeze(keypch_final(1,28,i)),squeeze(keypch_final(2,28,i)),squeeze(keypch_final(3,28,i)),...
%         matrixchild(7,i),matrixchild(8,i),matrixchild(9,i),matrixchild(10,i));                
% end

patientlabeldirectory = strcat(foldernamesp,'patientdata.txt');
%therapistlabeldirectory = strcat(direct,'/',patient,'/therapistdata.txt');

fileID = fopen(patientlabeldirectory);
H = textscan(fileID,'%f %s');
patientlabeldirectory
fclose(fileID);
keySet = {'Nowhere','kinect','robot','poster_dx','poster_sx','poster_dietro','altra_p'};
valueSet = [0 1 2 3 4 5 6];
M = containers.Map(keySet,valueSet);
for ihre=1:size(H{1,2},1)
    H{1,2}{ihre}
    if strcmp(H{1,2}{ihre},'toy')
        labels_fin(ihre) = 0;
    elseif strcmp(H{1,2}{ihre},'therapist') %therapist is not in the labels
        labels_fin(ihre) = 6;
    else
        labels_fin(ihre)=M(H{1,2}{ihre});
    end
end
tempos_labels=H{1,1};
for im=1:length(tempos_labels)
    m(im) = find(temposf==tempos_labels(im));
end
labels_all=ones(size(temposf))*-1;
labels_all(m)=labels_fin;

label_file = fopen(strcat(foldernamesp, '\labels_ch.txt'),'w');
j=1;
for i = 1:size(labels_all,2)
    if ~isnan(keypch_final(1,28,i))
        fprintf(label_file,'%d\n',labels_all(i));
        if ~isnan(labels_all(i))
            j
            j=j+1;
        end
    end
end
fclose(label_file)



 

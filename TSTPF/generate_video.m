% function generate_video(Iend,insertnum)

    savePath = '/Users/beixinzhu/Desktop/movie_result/television2/';
    imagePath = '/Users/beixinzhu/Desktop/movie_result/television2/';
%     savePath = '/Users/beixinzhu/Documents/dataset/';
%     imagePath = '/Users/beixinzhu/Documents/dataset/';

    writerObj = VideoWriter([savePath,'movie_tele_occ.avi']); % Name it.
    writerObj.FrameRate = 10; % How many frames per second - change this to make your video look nice
    open(writerObj);

    for i= 1:284
%         imname = sprintf([imagePath,'log000000%.4d.png'],i);
        imname = sprintf([imagePath,'frame_%.4d.png'],i);
        fprintf('i = %d\n',i);
        I = imread(imname);
        imshow(I);
        drawnow;
%         frame = getframe(gcf); 
        writeVideo(writerObj,I);
    end

    close(writerObj);

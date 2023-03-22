
<main>
    <div style:position="absolute" style:top=0px style:left=0px style:width={width}px style:height={height}px style:background-color="black">
      <div class="bar" style:background-color="#002233" style:position="absolute" style:width={barWidth}px style:height={barHeight}px style:left={barXCoordinate}px style:bottom={barYCoordinate}px>
        <input type="image" on:click={() => console.log("cool")} 
        src="/statistics.svg" width={iconSize}px style:position="absolute" style:top={firstBarElementYCoordinate + barElementYOffset*0}px style:left={firstBarElementXCoordinate + barElementXOffset*0}px/>
        <input type="image" on:click={() => {drawingBuffer.loadPixels(); predict_number(drawingBuffer.pixels);}} 
        src="/statistics.svg" width={iconSize}px style:position="absolute" style:top={firstBarElementYCoordinate + barElementYOffset*1}px style:left={firstBarElementXCoordinate + barElementXOffset*1}px/>
        <input type="image" on:click={() => console.log("cool")} 
        src="/statistics.svg" width={iconSize}px style:position="absolute" style:top={firstBarElementYCoordinate + barElementYOffset*2}px style:left={firstBarElementXCoordinate + barElementXOffset*2}px/>
      </div>
      <div style:position="absolute" style:width={canvasSize}px style:height={canvasSize}px style:left={canvasXCoordinate}px style:bottom={canvasYCoordinate}px>
        <P5 {sketch}/>
      </div>
    </div>
  </main>
  
  
  <script>


    export let width;
    export let height;

    let verticalBarWidth = 50;
    let horizontalBarHeight = 50;

    let barWidth;
    let barHeight;
    let canvasSize;

    let canvasXCoordinate;
    let canvasYCoordinate;

    let barXCoordinate;
    let barYCoordinate;

    let bufferWidth = 1000;
    let bufferHeight = 1000;

    let iconSize;
    let iconDownscaleFactor = 0.8;

    let firstBarElementXCoordinate;
    let firstBarElementYCoordinate;

    let barElementXOffset;
    let barElementYOffset;

    $: if(width<height-horizontalBarHeight){ //Vertical mode
        barWidth = width;
        barHeight = horizontalBarHeight;

        canvasSize = Math.min(width, height);

        canvasXCoordinate = 0;
        canvasYCoordinate = (height-canvasSize)/2.;

        barXCoordinate = 0;
        barYCoordinate = (height-canvasSize)/2 - horizontalBarHeight;

        iconSize = horizontalBarHeight*iconDownscaleFactor;

        firstBarElementXCoordinate = 0.2*iconSize;
        firstBarElementYCoordinate = horizontalBarHeight*(1-iconDownscaleFactor)/2;

        barElementXOffset = iconSize*1.2;
        barElementYOffset = 0;
    }
    else if (width < height + verticalBarWidth){ //Transition mode
        barWidth = width;
        barHeight = horizontalBarHeight;

        canvasSize = Math.min(width, height-horizontalBarHeight);

        canvasXCoordinate = (width-canvasSize)/2
        canvasYCoordinate = horizontalBarHeight;

        barXCoordinate = 0;
        barYCoordinate = 0;

        iconSize = horizontalBarHeight*iconDownscaleFactor;

        firstBarElementXCoordinate = 0.2*iconSize;
        firstBarElementYCoordinate = horizontalBarHeight*(1-iconDownscaleFactor)/2;

        barElementXOffset = iconSize*1.2;
        barElementYOffset = 0;
    }
    else{ //Landscape mode
        barHeight = height;
        barWidth = verticalBarWidth;

        canvasSize = Math.min(width, height);
        
        canvasXCoordinate = (width-canvasSize)/2;
        canvasYCoordinate = 0;

        barXCoordinate = (width - canvasSize)/2 + canvasSize;
        barYCoordinate = 0;

        iconSize = verticalBarWidth*iconDownscaleFactor;

        firstBarElementXCoordinate = verticalBarWidth*(1-iconDownscaleFactor)/2;
        firstBarElementYCoordinate = 0.2*iconSize;

        barElementXOffset = 0;
        barElementYOffset = iconSize*1.2;
    }


    

    import P5 from 'p5-svelte';
    import App from '../App.svelte';
  
  
    var drawingBuffer;
  
    const sketch = (p5) => {
      p5.draw = () => {
        if (p5.mouseIsPressed) {
          drawingBuffer.fill(0);
          drawingBuffer.ellipse(p5.mouseX/canvasSize*bufferWidth, p5.mouseY/canvasSize*bufferHeight, 80, 80);
        } 
  
        p5.image(drawingBuffer, 0, 0, canvasSize, canvasSize);
      };
  
  
      p5.setup = () => {
        p5.createCanvas(canvasSize, canvasSize);
        p5.background(200, 0, 100);
        drawingBuffer = p5.createGraphics(bufferWidth, bufferHeight);
        drawingBuffer.background(255, 255, 255);
      };
  
      
  
      p5.windowResized= () => {
        p5.resizeCanvas(canvasSize, canvasSize, true);
      }
    };
  
  
    function predict_number(pixels){
      console.log("Cool")
      console.log(pixels)
    }

  </script>
  
  
  
  
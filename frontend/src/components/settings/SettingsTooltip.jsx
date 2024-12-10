import React from 'react';

const SettingsTooltip = ({ tooltipIndex, tooltipVisible, setTooltipVisible, children }) => {
    return (
        <span className="relative ml-1 text-orange-400 hover:text-orange-600 cursor-pointer"
              onClick={(e) => {
                  e.stopPropagation(); // Prevents the event from bubbling up to parent elements
                  setTooltipVisible(tooltipIndex !== tooltipVisible() ? tooltipIndex : 0);
              }}>
            (?)
            {tooltipVisible() === tooltipIndex && (
                <span className="absolute w-60 -left-20 top-3 p-2 mt-2 text-sm text-white bg-black rounded-md shadow-lg z-10">
                    {children}
                </span>
            )}
        </span>
    );
};

export default SettingsTooltip;
/* eslint-disable */
/* tslint:disable */
import * as React from 'react';
export interface PaymentsProps extends React.SVGAttributes<SVGElement> {
size?: string | number;
}
const Payments: React.FC<PaymentsProps> = ({size, ...props}) => (
  <svg viewBox="0 0 40 40" fill="currentColor" width={ size || "40" } height={ size || "40" } {...props}>
    <path d="M30 10H10.75C10.286 10.0005 9.84122 10.1851 9.51315 10.5131C9.18508 10.8412 9.00053 11.286 9 11.75V28.375C9 28.5242 9.03813 28.6708 9.11076 28.8011C9.18339 28.9314 9.28812 29.0409 9.415 29.1193C9.54188 29.1977 9.6867 29.2424 9.83571 29.2491C9.98471 29.2558 10.133 29.2243 10.2664 29.1576L13.375 27.6033L16.4836 29.1576C16.6052 29.2183 16.7391 29.25 16.875 29.25C17.0109 29.25 17.1448 29.2183 17.2664 29.1576L20.375 27.6033L23.4836 29.1576C23.6052 29.2184 23.7391 29.25 23.875 29.25C24.0109 29.25 24.1448 29.2184 24.2664 29.1576L27.375 27.6033L30.4836 29.1576C30.617 29.2243 30.7653 29.2558 30.9143 29.2491C31.0633 29.2424 31.2081 29.1977 31.335 29.1193C31.4619 29.0409 31.5666 28.9314 31.6392 28.8011C31.7119 28.6708 31.75 28.5242 31.75 28.375V11.75C31.7495 11.286 31.5649 10.8412 31.2369 10.5131C30.9088 10.1851 30.464 10.0005 30 10ZM26.0625 21.375H14.6875C14.4554 21.375 14.2329 21.2828 14.0688 21.1187C13.9047 20.9546 13.8125 20.7321 13.8125 20.5C13.8125 20.2679 13.9047 20.0454 14.0688 19.8813C14.2329 19.7172 14.4554 19.625 14.6875 19.625H26.0625C26.2946 19.625 26.5171 19.7172 26.6812 19.8813C26.8453 20.0454 26.9375 20.2679 26.9375 20.5C26.9375 20.7321 26.8453 20.9546 26.6812 21.1187C26.5171 21.2828 26.2946 21.375 26.0625 21.375ZM26.0625 17.875H14.6875C14.4554 17.875 14.2329 17.7828 14.0688 17.6187C13.9047 17.4546 13.8125 17.2321 13.8125 17C13.8125 16.7679 13.9047 16.5454 14.0688 16.3813C14.2329 16.2172 14.4554 16.125 14.6875 16.125H26.0625C26.2946 16.125 26.5171 16.2172 26.6812 16.3813C26.8453 16.5454 26.9375 16.7679 26.9375 17C26.9375 17.2321 26.8453 17.4546 26.6812 17.6187C26.5171 17.7828 26.2946 17.875 26.0625 17.875Z"
      fill="#fff" />
  </svg>
);
Payments.displayName = 'Payments';
export default Payments;
/* tslint:enable */
/* eslint-enable */